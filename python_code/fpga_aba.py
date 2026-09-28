"""Ubuntu XDMA 批量 ABA：只访问计算板卡，不连接机器人。

协议与现有 run_aba_xrt.py 相同；句柄常驻，不启动子进程、不读写中间文本。
一个进程独占板卡。出错后停止，绝不在 FPGA 仍运行时覆盖输入或自动重试。
"""
from __future__ import annotations

import os
import struct
import time

import numpy as np


def pack_inputs(samples):
    """每行 q[6], dq[6], tau[6]，转换为小端有符号定点数。"""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 18 or not 1 <= len(samples) <= 1000:
        raise ValueError("ABA requires 1..1000 rows, each with 18 values")
    if not np.isfinite(samples).all():
        raise ValueError("ABA input contains NaN/Inf")
    scale = np.array([2.0**28] * 12 + [2.0**24] * 6)
    raw = np.rint(samples * scale)
    if np.any(raw < -(2**31)) or np.any(raw > 2**31 - 1):
        raise ValueError("Q4.28/Q8.24 input overflow; refusing to clip/wrap")
    return raw.astype("<i4").tobytes()


def unpack_outputs(payload, count):
    if len(payload) != count * 24:
        raise RuntimeError("Incomplete FPGA output")
    raw = np.frombuffer(payload, dtype="<i4").reshape(count, 6)
    if np.any(raw == -(2**31)) or np.any(raw == 2**31 - 1):
        raise RuntimeError("FPGA Q16.16 output saturated")
    # 内部运算饱和未必反映为最终 int32 极值，还必须通过动力学精度校验。
    return raw.astype(float) / 2.0**16


class XdmaABA:
    """不可跨线程共享。timeout 约束轮询，不保证能中断阻塞的驱动系统调用。"""

    def __init__(self, h2c="/dev/xdma0_h2c_0", c2h="/dev/xdma0_c2h_0",
                 base=0xC0000000, timeout=0.1, poll_interval=0.0):
        if os.name != "posix":
            raise RuntimeError("FPGA backend requires Ubuntu/Linux XDMA devices")
        if not np.isfinite([timeout, poll_interval]).all() or timeout <= 0 or poll_interval < 0:
            raise ValueError("Invalid timeout/poll interval")
        import fcntl
        self.base, self.timeout, self.poll_interval = base, timeout, poll_interval
        self.h2c = self.c2h = None
        self.failed = False
        self.last_timing = {}
        try:
            self.h2c = os.open(h2c, os.O_WRONLY)
            # 协作式锁：其他使用本类的进程不能同时写；旧脚本不遵守此锁。
            fcntl.flock(self.h2c, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.c2h = os.open(c2h, os.O_RDONLY)
            status = self._status()
            if status not in (0, 3):
                raise RuntimeError(f"FPGA busy/error (status={status}); reset/check manually")
            self._write(0x80, struct.pack("<I", 0))
            self._wait(0)
        except Exception:
            self.close()
            raise

    def _write(self, offset, payload):
        view = memoryview(payload)
        done = 0
        while done < len(view):
            count = os.pwrite(self.h2c, view[done:], self.base + offset + done)
            if count <= 0:
                raise RuntimeError("XDMA write returned no bytes")
            done += count

    def _read(self, offset, size):
        chunks, done = [], 0
        while done < size:
            chunk = os.pread(self.c2h, size - done, self.base + offset + done)
            if not chunk:
                raise RuntimeError("XDMA read returned no bytes")
            chunks.append(chunk)
            done += len(chunk)
        return b"".join(chunks)

    def _status(self):
        return struct.unpack("<I", self._read(0x84, 4))[0]

    def _wait(self, target):
        deadline = time.perf_counter() + self.timeout
        while True:
            status = self._status()
            if status == target:
                return
            if status not in (0, 1, 2, 3):
                raise RuntimeError(f"FPGA invalid/error status={status}")
            if time.perf_counter() >= deadline:
                raise TimeoutError(f"FPGA timeout, status={status}; no automatic retry")
            if self.poll_interval:
                time.sleep(self.poll_interval)

    def compute(self, samples):
        if self.failed:
            raise RuntimeError("Previous FPGA transaction failed; restart only after inspection")
        t0 = time.perf_counter()
        payload = pack_inputs(samples)
        t1 = time.perf_counter()
        try:
            if self._status() != 0:
                raise RuntimeError("FPGA is not idle; refusing to overwrite inputs")
            self._write(0x100, payload)
            self._write(0x88, struct.pack("<I", len(samples)))
            t2 = time.perf_counter()
            self._write(0x80, struct.pack("<I", 1))
            self._wait(3)
            t3 = time.perf_counter()
            result = unpack_outputs(self._read(0x12000, len(samples) * 24), len(samples))
            t4 = time.perf_counter()
            self._write(0x80, struct.pack("<I", 0))
            self._wait(0)  # 下一批开始前，确认 done 已清除，防止读取上一批结果。
            t5 = time.perf_counter()
        except Exception:
            self.failed = True
            raise
        self.last_timing = dict(pack_ms=(t1-t0)*1000, h2c_ms=(t2-t1)*1000,
                                command_ms=(t3-t2)*1000, c2h_ms=(t4-t3)*1000,
                                rearm_ms=(t5-t4)*1000, batch_ms=(t5-t0)*1000)
        return result

    def close(self):
        # 不通过写寄存器来取消未完成任务；现有 RTL 没有安全取消协议。
        for name in ("h2c", "c2h"):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)


class PinocchioABA:
    """CPU 对照后端；gravity=9.81 模拟当前 FPGA，gravity=0 对照原 MPC。"""

    def __init__(self, urdf, gravity=9.81):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf))
        if self.model.nq != 6 or self.model.nv != 6:
            raise ValueError("Expected a six-revolute-joint fixed-base robot")
        self.model.gravity.linear[:] = [0, 0, -gravity]
        self.model.gravity.angular[:] = 0
        self.data = self.model.createData()
        self.last_timing = {}

    def compute(self, samples):
        t0 = time.perf_counter()
        result = np.array([self.pin.aba(self.model, self.data, s[:6], s[6:12], s[12:]).copy()
                           for s in samples])
        self.last_timing = {"batch_ms": (time.perf_counter()-t0)*1000}
        return result

    def reference_torque(self, states, accelerations):
        return np.array([self.pin.rnea(self.model, self.data, x[:6], x[6:], a).copy()
                         for x, a in zip(states, accelerations)])

    def close(self):
        pass
