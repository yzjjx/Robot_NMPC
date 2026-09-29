r"""FPGA 控制器 + MuJoCo；默认 mock，不打开 FPGA 设备，也不连接机器人。

复用 v3 的 prepare/feedback 仿真循环：FPGA 参与准备阶段的 ABA 计算，
当前实现的解析导数和 QP 仍在 CPU 上计算。mock 耗时不代表实际 FPGA 耗时。

真实 FPGA 仿真（不发送机器人指令）：
    python3 track_mpc_fpga.py --backend fpga --allow-hardware
仅检查 ABA 通信，输入每行 q[6], dq[6], tau[6]，最多1000组：
    python3 track_mpc_fpga.py --backend fpga --allow-hardware \
        --check-protocol --aba-input aba_input.txt
仅当确认上一批结果已不需要时，追加 --clear-done；绝不清除运行中/错误状态。
"""

import argparse
import os
from pathlib import Path
import stat
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

TRAJECTORY = ROOT / "data_in" / "circle_R200_joint_trajectory_SR4_V50.txt"
HEADLESS = False
DURATION = None
CONTACTS = False
VIEWER_HZ = 60.0
REALTIME = False
MODEL_PERIOD = 0.01
FEEDBACK_RATIO = 2
HORIZON = 25
TORQUE_LIMIT = 10.0      # 当前 FPGA 数据范围要求不超过 10 N·m。
INTEGRATION_STEP = 0.001

BACKEND = "mock"       # "cpu"：CPU 对照；"mock"：软件设备；"fpga"：真实板卡。
ALLOW_HARDWARE = False  # 真实板卡需要主动改为 True，并配置 Linux XDMA 设备路径。
H2C = "/dev/xdma0_h2c_0"
C2H = "/dev/xdma0_c2h_0"
OUTPUT_DIR = ROOT / "data_out" / f"mpc_fpga_{BACKEND}"

# AXI字节地址，不是HLS AXI-Lite寄存器；与当前aba_ctrl_top.v一致。
AXI_BASE = 0xC0000000
COMMAND, STATUS, COUNT = 0x80, 0x84, 0x88
INPUT, OUTPUT = 0x100, 0x12000
MAX_BATCH = 1000
POLL_INTERVAL = 0.00001
TIMEOUT = 1.0
STATUS_NAMES = {0: "idle", 1: "loading", 2: "running", 3: "done", 4: "error"}


def pack_aba_inputs(values):
    """每组72字节：q/dq为Q4.28，tau为Q8.24；禁止溢出/静默限幅。"""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 18 or not 1 <= len(values) <= MAX_BATCH:
        raise ValueError("ABA 输入必须为 (1..1000, 18)，每行 q[6], dq[6], tau[6]。")
    limits = np.r_[np.full(6, np.pi), np.full(6, 2.0), np.full(6, 10.0)]
    if not np.isfinite(values).all() or np.any(np.abs(values) > limits):
        raise ValueError("ABA 输入非有限值或超域：|q|≤π，|dq|≤2，|tau|≤10。")
    scale = np.r_[np.full(12, 2.0**28), np.full(6, 2.0**24)]
    words = np.rint(values * scale)  # 与run_aba_xrt.py相同，最近偶数舍入。
    if np.any(words < -(2**31)) or np.any(words > 2**31-1):
        raise ValueError("ABA 输入定点编码溢出。")
    return words.astype("<i4").tobytes(order="C")


def decode_aba_outputs(data, count):
    """每组24字节，ddq[6]；不能把这些位当作IEEE float32。"""
    if not 1 <= count <= MAX_BATCH or len(data) != count * 24:
        raise ValueError(f"ABA 输出字节数不符：实际{len(data)}，期望{count * 24}。")
    words = np.frombuffer(data, dtype="<i4").reshape(count, 6)
    if np.any((words == -(2**31)) | (words == 2**31-1)):
        raise RuntimeError("ABA 输出出现Q16.16极限编码，拒绝使用疑似饱和结果。")
    return words.astype(np.float64) / 65536.0


class AbaMailbox:
    """仅用于启动握手/独立读回检查；MPC批处理仍由C++后端完成。

    同一时刻只能有一个XDMA客户端。用户态超时不能中断卡在驱动内的I/O。
    """

    def __init__(self, h2c, c2h):
        if not sys.platform.startswith("linux"):
            raise RuntimeError("真实 XDMA 访问需要 Ubuntu/Linux。")
        import fcntl
        self.h2c = self.c2h = None
        self.failed = False
        try:
            self.h2c = os.open(h2c, os.O_WRONLY | os.O_CLOEXEC)
            if not stat.S_ISCHR(os.fstat(self.h2c).st_mode):
                raise RuntimeError("H2C必须是XDMA字符设备，不能是普通文件。")
            fcntl.flock(self.h2c, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.c2h = os.open(c2h, os.O_RDONLY | os.O_CLOEXEC)
            if not stat.S_ISCHR(os.fstat(self.c2h).st_mode):
                raise RuntimeError("C2H必须是XDMA字符设备。")
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()  # 出错时仅关闭设备，绝不自动复位/重发。

    def close(self):
        for name in ("c2h", "h2c"):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    def read(self, offset, size):
        data = bytearray()
        deadline = time.monotonic() + TIMEOUT
        while len(data) < size:
            if time.monotonic() >= deadline:
                raise TimeoutError("XDMA读超时。")
            try:
                part = os.pread(self.c2h, size-len(data), AXI_BASE+offset+len(data))
            except InterruptedError:
                continue
            if not part:
                raise RuntimeError("XDMA短读：设备返回0字节。")
            data.extend(part)
        return bytes(data)

    def write(self, offset, data):
        view = memoryview(data)
        sent = 0
        deadline = time.monotonic() + TIMEOUT
        while sent < len(view):
            if time.monotonic() >= deadline:
                raise TimeoutError("XDMA写超时。")
            try:
                count = os.pwrite(self.h2c, view[sent:], AXI_BASE+offset+sent)
            except InterruptedError:
                continue
            if count <= 0:
                raise RuntimeError("XDMA短写：设备未接收数据。")
            sent += count

    def read_word(self, offset):
        return struct.unpack("<I", self.read(offset, 4))[0]

    def write_word(self, offset, value):
        # 只写低32位command；绝不能覆盖相邻+0x84的FPGA状态。
        self.write(offset, struct.pack("<I", value))

    def wait_status(self, expected):
        deadline = time.monotonic() + TIMEOUT
        while True:
            status = self.read_word(STATUS)
            if status not in STATUS_NAMES or status == 4:
                raise RuntimeError(f"FPGA status={status}，停止；请检查板卡，不自动清错。")
            if status == expected:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"等待status={expected}超时，最后status={status}。")
            time.sleep(POLL_INTERVAL)

    def ensure_idle(self, clear_done=False):
        if self.failed:
            raise RuntimeError("本客户端此前通信失败，不能重用；请检查板卡状态。")
        command, status = struct.unpack("<II", self.read(COMMAND, 8))
        if command == 0 and status in (0, 3):
            self.wait_status(0)  # 上一进程已写clear，允许RTL完成idle握手。
        elif command == 1 and status == 3 and clear_done:
            print("丢弃上一批未确认的完成状态：command=0，等待idle。")
            self.write_word(COMMAND, 0)
            self.wait_status(0)
        else:
            raise RuntimeError(
                f"FPGA未就绪：command={command}, status={status} "
                f"({STATUS_NAMES.get(status, 'unknown')})。"
                "仅当上一批已完成且结果不再需要时使用 --clear-done；"
                "运行中/错误状态不会自动复位。")

    def calculate(self, values):
        payload = pack_aba_inputs(values)
        count = len(payload) // 72
        try:
            self.ensure_idle()
            self.write(INPUT, payload)
            self.write_word(COUNT, count)
            self.write_word(COMMAND, 1)
            # 轮询可能跳过loading/running，读到done即可；前置idle排除了旧done。
            self.wait_status(3)
            result = decode_aba_outputs(self.read(OUTPUT, count * 24), count)
            self.write_word(COMMAND, 0)  # 保存本批结果后才清除完成状态。
            self.wait_status(0)
            return result
        except Exception:
            self.failed = True
            raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "mock", "fpga"), default=BACKEND)
    parser.add_argument("--allow-hardware", action="store_true", default=ALLOW_HARDWARE)
    parser.add_argument("--h2c", default=H2C)
    parser.add_argument("--c2h", default=C2H)
    parser.add_argument("--clear-done", action="store_true",
                        help="明确丢弃上一次已完成批次；不清除running/error")
    parser.add_argument("--check-protocol", action="store_true",
                        help="只提交ABA输入并读取ddq，不加载MPC扩展/MuJoCo")
    parser.add_argument("--aba-input", type=Path, help="每行18列的q/dq/tau文本，最多1000组")
    parser.add_argument("--module-dir", type=Path, help="rokae_mpc_fpga.so所在目录")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.backend == "fpga" and not args.allow_hardware:
        parser.error("真实FPGA需要 --backend fpga --allow-hardware。")
    if (args.check_protocol or args.clear_done or args.aba_input) and args.backend != "fpga":
        parser.error("协议检查/clear-done/aba-input只适用于fpga后端。")
    if args.aba_input and not args.check_protocol:
        parser.error("--aba-input仅用于--check-protocol，不替换MPC的参考轨迹。")
    return args


def hardware_startup(args):
    """在C++接管设备前完成握手；检查模式额外显示真实ABA返回值。"""
    values = None
    if args.check_protocol:
        values = (np.loadtxt(args.aba_input, ndmin=2) if args.aba_input else
                  np.zeros((1, 18)))
        pack_aba_inputs(values)  # 打开设备前先检查输入。
    print(f"XDMA: {args.h2c} / {args.c2h}; input=0x{AXI_BASE+INPUT:X}, "
          f"output=0x{AXI_BASE+OUTPUT:X}")
    print("q/dq=Q4.28, tau=Q8.24, ddq=Q16.16；均为小端有符号int32。")
    with AbaMailbox(args.h2c, args.c2h) as mailbox:
        mailbox.ensure_idle(clear_done=args.clear_done)
        if values is not None:
            result = mailbox.calculate(values)
            print(f"本批FPGA ddq：{len(result)}组，单位rad/s²；前5组：")
            print(result[:5])
            print("读回成功不等于动力学模型正确；完整MPC仍必须通过模型一致性检查。")
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                path = args.output_dir / "aba_protocol_output.txt"
                np.savetxt(path, result, fmt="%.10g")
                print(f"保存输出：{path}")
    # 释放Python设备/锁，再由C++独占设备；两套客户端不并行运行。


def validate_backend(controller, reference):
    """输入代表性状态和力矩，与 CPU 比较 ABA 输出；不一致时由扩展抛出异常。"""
    limits = np.r_[np.full(6, np.pi), np.full(6, 2.0)]
    if np.any(np.abs(reference) > limits):
        raise ValueError("参考轨迹超出当前 FPGA 范围：|q|≤π rad，|dq|≤2 rad/s。")

    # 起点附近每个 q/dq/tau 分量分别正负扰动，保证 18 个输入维度均有变化。
    states = np.tile(reference[0], (37, 1))
    torques = np.zeros((37, 6))
    for j in range(18):
        if j < 12:
            states[1+2*j, j] += 0.02
            states[2+2*j, j] -= 0.02
        else:
            torques[1+2*j, j-12] = min(0.1, TORQUE_LIMIT)
            torques[2+2*j, j-12] = -min(0.1, TORQUE_LIMIT)
    # 仅把测试扰动限制在有效域内，不修改用户的轨迹。
    states = np.clip(states, -limits, limits)

    # 再沿整条参考轨迹取样，并检查正负力矩。这仍是有限样本检查，
    # 不是整个工作空间的正确性证明；后端运行时也会检查每次 ABA 输入范围。
    indices = np.linspace(0, len(reference)-1, min(64, len(reference)), dtype=int)
    sampled = reference[indices]
    states = np.vstack((states, sampled, sampled, sampled))
    test_tau = np.ones((len(sampled), 6)) * TORQUE_LIMIT
    torques = np.vstack((torques, np.zeros_like(test_tau), test_tau, -test_tau))
    controller.validate_backend(states, torques, absolute_tolerance=0.05,
                                relative_tolerance=0.002)
    print(f"ABA 模型一致性检查通过：{len(states)} 组输入。")


def main():
    args = parse_args()
    if args.check_protocol:
        hardware_startup(args)
        return
    if args.module_dir:
        if not args.module_dir.is_dir():
            raise ValueError(f"模块目录不存在：{args.module_dir}")
        sys.path.insert(0, str(args.module_dir.resolve()))
    try:
        import rokae_mpc_fpga as mpc
    except ImportError as error:
        raise ImportError(
            "无法加载rokae_mpc_fpga扩展。请用 --module-dir 指向Ubuntu已构建的"
            "build_fpga_v3/python目录，并确认Python版本/依赖匹配。"
            "只有rokae_mpc_v3.so不能代替FPGA扩展；--check-protocol不需要此扩展。"
        ) from error
    from track_mpc_v3 import make_config, run_simulation

    if not np.isfinite(TORQUE_LIMIT) or not 0 < TORQUE_LIMIT <= 10:
        raise ValueError("FPGA 示例要求 0 < TORQUE_LIMIT <= 10 N·m。")
    config = make_config(mpc, MODEL_PERIOD, FEEDBACK_RATIO, HORIZON,
                         TORQUE_LIMIT, INTEGRATION_STEP, REALTIME)
    print(f"MPC扩展：{mpc.__file__}；请求后端：{args.backend}")
    if args.backend == "fpga":
        hardware_startup(args)
    controller = mpc.MPCController(str(ROOT / "urdf" / "ROKAE_SR4.urdf"), config,
                                   backend=args.backend, allow_hardware=args.allow_hardware,
                                   h2c=args.h2c, c2h=args.c2h)
    if args.backend == "fpga" and controller.backend_name != "xdma_hardware":
        raise RuntimeError(f"未加载真实XDMA后端：{controller.backend_name}；请检查扩展版本。")
    if args.backend == "mock":
        print("mock 是软件后端，不能用其耗时评价真实 FPGA 加速效果。")
    output_dir = args.output_dir or (OUTPUT_DIR if args.backend == BACKEND else
                                    ROOT / "data_out" / f"mpc_fpga_{args.backend}")
    run_simulation(controller, config, trajectory=TRAJECTORY, headless=HEADLESS,
                   duration=DURATION, contacts=CONTACTS, viewer_hz=VIEWER_HZ,
                   realtime=REALTIME, output_dir=output_dir,
                   backend_check=validate_backend if args.backend != "cpu" else None)


if __name__ == "__main__":
    main()
