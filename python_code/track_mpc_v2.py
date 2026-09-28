"""简单的同步 MuJoCo 仿真。仿真在求解期间暂停，不代表实时/实物实验。"""
import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
from rokae_mpc_v2 import MPCController

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--trajectory", type=Path,
                        default=ROOT / "data_in/circle_R200_joint_trajectory_SR4_V50.txt")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if not np.isfinite(args.period) or args.period <= 0 or not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("period and duration must be finite and positive")
    if args.horizon < 1:
        parser.error("horizon must be positive")

    model = mujoco.MjModel.from_xml_path(str(ROOT / "xml/ROKAE_SR4.XML"))
    dt = float(model.opt.timestep)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("XML timestep must be finite and positive")
    control_steps = round(args.period / dt)
    if control_steps < 1 or not np.isclose(control_steps*dt, args.period, rtol=0, atol=1e-9):
        parser.error("period must be an integer multiple of the XML timestep")
    if (model.nq, model.nv, model.nu) != (6, 6, 6):
        raise ValueError("Expected six joints and six torque actuators")
    # 保持旧仿真约定：重力由底层补偿，关闭接触；不是实物安全模型。
    model.opt.gravity[:] = 0
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    controller = MPCController(str(ROOT / "urdf/ROKAE_SR4.urdf"), args.period, args.horizon)

    raw = np.loadtxt(args.trajectory, skiprows=1)
    if raw.ndim != 2 or raw.shape[1] != 13 or len(raw) < 3 or not np.isfinite(raw).all():
        raise ValueError("Expected time + six q + six dq columns, at least three rows")
    timestamps = raw[:, 0] - raw[0, 0]
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Reference timestamps must be strictly increasing")
    acceleration = np.gradient(raw[:, 7:13], timestamps, axis=0, edge_order=2)

    def reference_at(query):
        states = np.column_stack([np.interp(query, timestamps, raw[:, j]) for j in range(1, 13)])
        acc = np.column_stack([np.interp(query, timestamps, acceleration[:, j]) for j in range(6)])
        states[query > timestamps[-1], 6:] = 0
        acc[query > timestamps[-1]] = 0
        return states, acc

    data = mujoco.MjData(model)
    data.qpos[:] = raw[0, 1:7]
    data.qvel[:] = raw[0, 7:13]
    mujoco.mj_forward(model, data)
    tau = np.zeros(6)
    rows, timing_rows = [], []
    steps = int(min(args.duration, timestamps[-1]) / dt)
    if steps < 1:
        raise ValueError("Requested simulation is shorter than one step")
    window = mujoco.viewer.launch_passive(model, data) if args.viewer else nullcontext(None)
    next_render = time.perf_counter()
    with window as viewer:
        for k in range(steps):
            if viewer is not None and not viewer.is_running():
                break
            if k % control_steps == 0:
                query = data.time + np.arange(args.horizon+1)*args.period
                refs, acc = reference_at(query)
                measured = np.r_[data.qpos, data.qvel]
                start = time.perf_counter()
                # 使用本次测量同步求解；失败即抛异常停止仿真，不采用过期结果。
                tau = controller.compute_control(measured, refs, acc[:-1])
                elapsed = 1000*(time.perf_counter()-start)
                t = controller.timing
                timing_rows.append([data.time, elapsed, t.reference_ms, t.dynamics_ms,
                                    t.matrices_ms, t.qp_ms, t.qp_iterations])
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Simulation became nonfinite")
            rows.append(np.r_[data.time, data.qpos, data.qvel, tau])
            if viewer is not None and time.perf_counter() >= next_render:
                viewer.sync()
                next_render = time.perf_counter()+1/60

    if not timing_rows:
        print("No control cycles completed.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    tracking = np.asarray(rows)
    np.savetxt(args.output / "tracking_v2.csv", tracking, delimiter=",",
               header="time,"+",".join(f"{s}{j}" for s in ("q", "dq", "tau") for j in range(1,7)), comments="")
    timing = np.asarray(timing_rows)
    np.savetxt(args.output / "timing_v2.csv", timing, delimiter=",",
               header="time,wall_ms,reference_ms,dynamics_ms,matrices_ms,qp_ms,qp_iterations", comments="")
    refs, _ = reference_at(tracking[:, 0])
    print("Joint RMSE [rad]:", np.sqrt(np.mean((tracking[:,1:7]-refs[:,:6])**2, axis=0)))
    print(f"Compute mean/P99/max [ms]: {timing[:,1].mean():.3f} / "
          f"{np.percentile(timing[:,1],99):.3f} / {timing[:,1].max():.3f}")
    print(f"Compute-budget misses: {np.count_nonzero(timing[:,1] >= 1000*args.period)}/{len(timing)}")
    print("Offline synchronous simulation; these numbers do not certify a real-time control rate.")


if __name__ == "__main__":
    main()
