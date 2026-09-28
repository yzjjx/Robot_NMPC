#!/usr/bin/env python3
"""Ubuntu 上的 FPGA-MPC 仿真入口。无 C++ 扩展依赖，不需要重新生成 bit。

默认 CPU 对照；--backend fpga 才访问 XDMA。此程序不发送真实机器人命令。
仿真同步推进，理想零计算延迟；测得的求解耗时用于评估控制周期预算，
不能将设定的仿真频率当成真实机器人已经达到的闭环频率。
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import time

# 小矩阵多线程调度可能比计算还慢。允许调用者通过环境变量覆盖。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

from fpga_aba import PinocchioABA, XdmaABA
from fpga_mpc import FpgaMPC, linearize_batch

ROOT = Path(__file__).resolve().parent.parent


class Reference:
    def __init__(self, path):
        raw = np.loadtxt(path, skiprows=1)
        if raw.ndim != 2 or raw.shape[1] != 13 or len(raw) < 3 or not np.isfinite(raw).all():
            raise ValueError("Reference must contain time,q[6],dq[6], with a header")
        self.t = raw[:, 0] - raw[0, 0]
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("Reference timestamps must strictly increase")
        self.x = raw[:, 1:]
        self.acc = np.gradient(self.x[:, 6:], self.t, axis=0, edge_order=2)

    def sample(self, times):
        times = np.atleast_1d(times)
        x = np.column_stack([np.interp(times, self.t, self.x[:, j],
                                      right=self.x[-1, j] if j < 6 else 0.0)
                             for j in range(12)])
        acc = np.column_stack([np.interp(times, self.t, self.acc[:, j], right=0.0)
                               for j in range(6)])
        return x, acc


def check_models(backend, raw_cpu, zero_cpu, reference, args):
    """启动时验证位流/URDF/重力/定点差分；不靠 NaN 检查判断结果正确。"""
    x, ddq = reference.sample(np.linspace(0, reference.t[-1], 12))
    u = zero_cpu.reference_torque(x, ddq)
    u = np.clip(u, -args.torque_limit, args.torque_limit)
    # 增加确定性的非零力矩测试，避免只在一条低负载轨迹上偶然吻合。
    rng = np.random.default_rng(20260831)
    test = np.concatenate((np.tile(x, (2, 1)),
                           np.vstack((u, rng.uniform(-0.8*args.torque_limit,
                                                      0.8*args.torque_limit, (12, 6))))), axis=1)
    actual, expected = backend.compute(test), raw_cpu.compute(test)
    error = np.abs(actual - expected)
    limit = args.check_atol + args.check_rtol*np.abs(expected)
    print(f"Raw ABA check: max_abs={error.max():.6g} rad/s^2, "
          f"RMSE={np.sqrt(np.mean(error**2)):.6g}, normalized_max={np.max(error/limit):.3f}")
    if not np.isfinite(actual).all() or np.any(error > limit):
        row, joint = np.unravel_index(np.argmax(error/limit), error.shape)
        raise RuntimeError(
            f"FPGA/URDF mismatch at probe {row}, joint {joint+1}: "
            f"FPGA={actual[row,joint]:g}, CPU={expected[row,joint]:g}. "
            "Check bitstream format, joint axes/offsets, inertia and gravity; do not bypass this check.")
    steps = (args.fd_q, args.fd_dq, args.fd_tau)
    f_hw, _, _, j_hw = linearize_batch(backend, x, u, args.period,
                                       args.fpga_gravity == "earth", steps)
    f_cpu, _, _, j_cpu = linearize_batch(zero_cpu, x, u, args.period, False, steps)
    acceleration_error = np.abs((f_hw[:, 6:] - f_cpu[:, 6:]) / args.period)
    cpu_acceleration = (f_cpu[:, 6:] - x[:, 6:]) / args.period
    if np.any(acceleration_error > args.check_atol + args.check_rtol*np.abs(cpu_acceleration)):
        raise RuntimeError("Gravity cancellation/zero-gravity ABA accuracy check failed")
    # 每个导数列按六关节向量的范数比较，近零导数用 1 作尺度下限。
    ratio = np.linalg.norm(j_hw-j_cpu, axis=1) / np.maximum(1.0, np.linalg.norm(j_cpu, axis=1))
    print("Jacobian check (normalized column error): "
          f"q={ratio[:, :6].max():.4g}, dq={ratio[:, 6:12].max():.4g}, "
          f"tau={ratio[:, 12:].max():.4g}")
    if not np.isfinite(j_hw).all() or ratio.max() > args.jac_tolerance:
        raise RuntimeError("Fixed-point finite differences are not accurate enough for MPC; "
                           "compare larger fd steps and investigate internal quantization")
    print("Startup checks passed on sampled points (not a full-domain certification).")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("cpu", "fpga"), default="cpu")
    p.add_argument("--period", type=float, default=0.01, help="MPC period / prediction step, seconds")
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--torque-limit", type=float, default=10.0,
                   help="Nm; default matches HLS documented test envelope, original MPC used 30")
    p.add_argument("--trajectory", type=Path,
                   default=ROOT / "data_in/circle_R200_joint_trajectory_SR4_V50.txt")
    p.add_argument("--urdf", type=Path, default=ROOT / "urdf/ROKAE_SR4.urdf")
    p.add_argument("--xml", type=Path, default=ROOT / "xml/ROKAE_SR4.XML")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data_out/fpga_mpc")
    p.add_argument("--viewer", action="store_true", help="Unpaced simulation viewer; off for timing")
    p.add_argument("--check-only", action="store_true", help="Check models without MuJoCo or MPC run")
    p.add_argument("--fpga-gravity", choices=("earth", "zero"), default="earth",
                   help="Gravity in the loaded bitstream; local HLS source uses earth")
    p.add_argument("--h2c", default="/dev/xdma0_h2c_0")
    p.add_argument("--c2h", default="/dev/xdma0_c2h_0")
    p.add_argument("--base-address", type=lambda s: int(s, 0), default=0xC0000000)
    p.add_argument("--timeout", type=float, default=0.1)
    p.add_argument("--poll-interval", type=float, default=0.0, help="Busy-poll by default (uses a CPU core)")
    p.add_argument("--fd-q", type=float, default=0.002)
    p.add_argument("--fd-dq", type=float, default=0.002)
    p.add_argument("--fd-tau", type=float, default=0.01)
    p.add_argument("--check-atol", type=float, default=0.1)
    p.add_argument("--check-rtol", type=float, default=0.02)
    p.add_argument("--jac-tolerance", type=float, default=0.2)
    args = p.parse_args()
    positive = [args.period, args.duration, args.torque_limit, args.timeout, args.fd_q,
                args.fd_dq, args.fd_tau, args.check_atol, args.check_rtol, args.jac_tolerance]
    if not np.isfinite(positive).all() or min(positive) <= 0:
        p.error("All periods, limits, finite-difference steps and tolerances must be positive")
    if not np.isfinite(args.poll_interval) or args.poll_interval < 0:
        p.error("poll-interval must be nonnegative")
    if args.torque_limit > 30 or args.fd_q >= 0.1 or args.fd_dq >= 0.1:
        p.error("Torque limit must be <=30; q/dq differences must be <0.1")
    max_horizon = 20 if args.fpga_gravity == "earth" else 27
    if not 1 <= args.horizon <= max_horizon:
        p.error(f"Single-batch horizon must be 1..{max_horizon}")
    return args


def run_simulation(controller, reference, args):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    if (model.nq, model.nv, model.nu) != (6, 6, 6):
        raise ValueError("Expected six MuJoCo joints and direct torque actuators")
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    dt = float(model.opt.timestep)
    stride = int(round(args.period/dt))
    if stride < 1 or not np.isclose(stride*dt, args.period, atol=1e-10, rtol=0):
        raise ValueError("MPC period must be an integer multiple of XML timestep")
    data = mujoco.MjData(model)
    initial, _ = reference.sample(0)
    data.qpos[:], data.qvel[:] = initial[0, :6], initial[0, 6:]
    mujoco.mj_forward(model, data)
    steps = int(min(args.duration, reference.t[-1])/dt)
    if steps < stride:
        raise ValueError("Duration shorter than one MPC interval")
    timing, records = [], []
    if args.viewer:
        import mujoco.viewer
        view = mujoco.viewer.launch_passive(model, data)
    else:
        view = nullcontext(None)
    start = time.perf_counter()
    next_draw = 0.0
    try:
        with view as viewer:
            for k in range(steps):
                if viewer is not None and not viewer.is_running():
                    break
                now = k*dt
                if k % stride == 0:
                    # 先测状态再同步求解；不沿用旧版本的后台迟到控制结果。
                    begin = time.perf_counter()
                    state = np.r_[data.qpos, data.qvel]
                    ref, ddq = reference.sample(now + np.arange(args.horizon+1)*args.period)
                    tau = controller.compute_control(state, ref, ddq[:-1])
                    data.ctrl[:] = tau
                    entry = dict(sim_time=now, **controller.last_timing)
                    entry["control_wall_ms"] = (time.perf_counter()-begin)*1000
                    timing.append(entry)
                mujoco.mj_step(model, data)
                if not np.isfinite(np.r_[data.qpos, data.qvel]).all() or not np.isclose(data.time, (k+1)*dt):
                    raise RuntimeError("MuJoCo instability/reset detected")
                records.append(np.r_[data.time, data.qpos, data.qvel, data.ctrl])
                wall = time.perf_counter()
                if viewer is not None and wall >= next_draw:
                    viewer.sync()
                    next_draw = wall + 1/60
    finally:
        # 使用唯一子目录保留每次实验，异常也保存已完成部分，绝不覆盖旧日志。
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / f"{args.backend}_{time.time_ns()}"
        output.mkdir()
        if records:
            np.savetxt(output / "states.csv", records, delimiter=",",
                       header="time," + ",".join(f"{s}{i}" for s in ("q", "dq", "tau") for i in range(1, 7)),
                       comments="")
        if timing:
            keys = list(timing[0])
            np.savetxt(output / "timing.csv", [[row[key] for key in keys] for row in timing],
                       delimiter=",", header=",".join(keys), comments="")
            print(f"First control call (includes QP setup): {timing[0]['control_wall_ms']:.3f} ms")
            measured = np.array([row["control_wall_ms"] for row in timing[1:]])
            if len(measured):
                print(f"Control wall time: mean={measured.mean():.3f}, "
                      f"p99={np.percentile(measured,99):.3f}, max={measured.max():.3f} ms")
                print(f"Budget overruns: {np.sum(measured > args.period*1000)}/{len(measured)} "
                      f"at target {1/args.period:.1f} Hz")
                print(f"Inverse mean compute time (not actual closed-loop Hz): {1000/measured.mean():.1f} Hz")
        if records:
            record = np.asarray(records)
            goal, _ = reference.sample(record[:, 0])
            err_deg = np.rad2deg(record[:, 1:7] - goal[:, :6])
            print(f"Tracking error: RMS={np.sqrt(np.mean(err_deg**2)):.5f} deg, "
                  f"max={np.abs(err_deg).max():.5f} deg")
        print(f"Experiment wall time (includes logging): {time.perf_counter()-start:.3f} s; logs: {output}")


def main():
    args = arguments()
    ref = Reference(args.trajectory)
    if np.any(np.abs(ref.x[:, :6]) >= np.pi-2*args.fd_q) or np.any(np.abs(ref.x[:, 6:]) >= 2-2*args.fd_dq):
        raise ValueError("Reference exceeds conservative HLS q/dq deployment envelope")
    zero = PinocchioABA(args.urdf, gravity=0)
    raw = PinocchioABA(args.urdf, gravity=9.81 if args.fpga_gravity == "earth" else 0)
    backend = (XdmaABA(args.h2c, args.c2h, args.base_address, args.timeout, args.poll_interval)
               if args.backend == "fpga" else raw)
    try:
        check_models(backend, raw, zero, ref, args)
        if args.check_only:
            return
        controller = FpgaMPC(backend, zero, args.period, args.horizon, args.torque_limit,
                             args.fpga_gravity == "earth", (args.fd_q, args.fd_dq, args.fd_tau))
        batch_size = args.horizon * (50 if args.fpga_gravity == "earth" else 37)
        print(f"Target={1/args.period:.1f} Hz; horizon={args.horizon*args.period:.3f} s; "
              f"ABA batch={batch_size}; torque limit=+/-{args.torque_limit:g} Nm")
        print("Ideal zero-delay, unpaced MuJoCo simulation; NOT a real-robot controller.")
        run_simulation(controller, ref, args)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
