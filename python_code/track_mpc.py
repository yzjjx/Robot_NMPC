"""1000 Hz 力矩输入 + 后台 MPC + 零阶保持，按真实时间运行 MuJoCo。"""

import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from rokae_mpc import MPCController

ROOT = Path(__file__).resolve().parent.parent
TRAJECTORY = ROOT / "data_in" / "circle_R200_joint_trajectory_SR4_V50.txt"
INPUT_DT = 0.001  # 向 MuJoCo 输入力矩的周期，1000 Hz。


def load_reference(dt, horizon):
    """按 MPC 的周期插值；原文件没有加速度，用速度对时间求导得到。"""
    raw = np.loadtxt(TRAJECTORY, skiprows=1)
    if raw.ndim != 2 or raw.shape[1] != 13 or len(raw) < 3:
        raise ValueError("轨迹文件需要时间、6 个位置、6 个速度，共 13 列。")
    if not np.isfinite(raw).all() or not np.all(np.diff(raw[:, 0]) > 0):
        raise ValueError("轨迹数据必须有限，且时间严格递增。")
    source_time = raw[:, 0] - raw[0, 0]
    acceleration = np.gradient(raw[:, 7:13], source_time, axis=0, edge_order=2)
    steps = int(np.ceil(source_time[-1] / dt))
    sample_time = np.arange(steps + horizon + 1) * dt

    q = np.column_stack([
        np.interp(sample_time, source_time, raw[:, j]) for j in range(1, 7)
    ])
    dq = np.column_stack([
        np.interp(sample_time, source_time, raw[:, j], right=0.0)
        for j in range(7, 13)
    ])
    ddq = np.column_stack([
        np.interp(sample_time, source_time, acceleration[:, j], right=0.0)
        for j in range(6)
    ])
    # 超过轨迹末尾时，位置保持最后一点，速度和加速度均为零。
    return np.column_stack((q, dq)), ddq, steps


def solve_next(controller, state, held_tau, states, accelerations):
    """后台任务：预测旧力矩保持一周期后的状态，计算下一更新时刻的力矩。"""
    start = time.perf_counter()
    predicted = controller.predict_state(state, held_tau, controller.timestep)
    tau = controller.compute_control(predicted, states, accelerations)
    finished = time.perf_counter()
    return tau, 1000 * (finished - start), finished


def save_results(rows, updates, output_dir, summary):
    """保存原始数据、误差统计和图片；关闭界面时也保留已完成的数据。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    if updates:
        with (output_dir / "mpc_updates.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(updates[0]))
            writer.writeheader()
            writer.writerows(updates)
        solve_times = np.array([item["solve_ms"] for item in updates])
        summary.update({
            "solve_mean_ms": float(np.mean(solve_times)),
            "solve_p95_ms": float(np.percentile(solve_times, 95)),
            "solve_max_ms": float(np.max(solve_times)),
            "solve_overrun_percent": float(100 * np.mean(
                solve_times > 1000 * summary["mpc_period_s"])),
        })
    if rows:
        result = np.asarray(rows)
        names = ["time_s"]
        for prefix in ("q_ref_rad", "q_rad", "dq_ref_rad_s", "dq_rad_s", "tau_Nm"):
            names += [f"{prefix}_{j+1}" for j in range(6)]
        names += ["solve_ms", "cycle_ms"]
        names += [f"{prefix}_{axis}" for prefix in ("tcp_ref_m", "tcp_m")
                  for axis in ("x", "y", "z")]
        names += ["torque_updated", "wall_lag_ms"]
        np.savetxt(output_dir / "tracking.csv", result, delimiter=",",
                   header=",".join(names), comments="")

        error_deg = np.rad2deg(result[:, 7:13] - result[:, 1:7])
        tcp_error_mm = 1000 * np.linalg.norm(result[:, 36:39] - result[:, 33:36], axis=1)
        dt_ms = 1000 * summary["control_dt_s"]
        summary.update({
            "samples": len(result),
            "simulated_time_s": float(result[-1, 0]),
            "joint_rmse_deg": np.sqrt(np.mean(error_deg**2, axis=0)).tolist(),
            "joint_max_abs_error_deg": np.max(np.abs(error_deg), axis=0).tolist(),
            "velocity_rmse_rad_s": np.sqrt(np.mean(
                (result[:, 19:25] - result[:, 13:19])**2, axis=0)).tolist(),
            "tcp_rmse_mm": float(np.sqrt(np.mean(tcp_error_mm**2))),
            "tcp_max_error_mm": float(np.max(tcp_error_mm)),
            "max_abs_torque_Nm": np.max(np.abs(result[:, 25:31]), axis=0).tolist(),
            "cycle_mean_ms": float(np.mean(result[:, 32])),
            "cycle_max_ms": float(np.max(result[:, 32])),
            "cycle_overrun_percent": float(100 * np.mean(result[:, 32] > dt_ms)),
            "max_wall_lag_ms": float(np.max(result[:, 40])),
            "wall_lag_over_1ms_percent": float(100 * np.mean(result[:, 40] > dt_ms)),
            "torque_updates_including_initial": int(np.sum(result[:, 39])),
        })

        import matplotlib
        matplotlib.use("Agg")  # 无界面模式下也能保存图片。
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(6, 2, figsize=(12, 15), sharex=True)
        for j in range(6):
            axes[j, 0].plot(result[:, 0], np.rad2deg(result[:, 1+j]), label="Reference")
            axes[j, 0].plot(result[:, 0], np.rad2deg(result[:, 7+j]), "--", label="MuJoCo")
            axes[j, 0].set_ylabel(f"Joint {j+1} (deg)")
            axes[j, 1].plot(result[:, 0], error_deg[:, j])
            axes[j, 1].set_ylabel("Error (deg)")
            for ax in axes[j]:
                ax.grid(True)
        axes[0, 0].legend()
        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(output_dir / "tracking.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
        for j, ax in enumerate(axes):
            ax.step(result[:, 0] - summary["control_dt_s"], result[:, 25+j], where="post")
            ax.set_ylabel(f"tau{j+1} (Nm)")
            ax.grid(True)
        axes[0].set_title("1000 Hz torque input with zero-order hold")
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(output_dir / "torque_hold.png", dpi=150)
        plt.close(fig)

        print("各关节位置 RMSE (deg)：", np.round(summary["joint_rmse_deg"], 6))
        print(f"tool_site 位置 RMSE：{summary['tcp_rmse_mm']:.4f} mm")
        print(f"仿真/实际运行时间：{summary['simulated_time_s']:.3f} / "
              f"{summary['wall_elapsed_s']:.3f} s，MPC 更新错过次数："
              f"{summary['mpc_deadline_misses']}")
    if updates:
        print(f"后台 MPC 平均/最大耗时：{summary['solve_mean_ms']:.3f} / "
              f"{summary['solve_max_ms']:.3f} ms")

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"结果：{output_dir.resolve()}，状态：{summary['status']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="不打开 MuJoCo 窗口")
    parser.add_argument("--duration", type=float, help="只运行前多少秒，默认完整轨迹")
    parser.add_argument("--contacts", action="store_true", help="保留 XML 网格的接触作用")
    parser.add_argument("--mpc-period", type=float, default=0.1, help="MPC 更新周期，默认 0.1 秒")
    parser.add_argument("--horizon", type=int, default=10, help="MPC 预测步数，默认 10")
    parser.add_argument("--output", type=Path, default=ROOT / "data_out" / "mpc_zoh_1000hz")
    args = parser.parse_args()
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration 必须为正数。")

    dt = INPUT_DT
    if not np.isfinite(args.mpc_period) or args.mpc_period < dt or args.horizon < 1:
        parser.error("MPC 周期必须至少为 1 ms，预测步数必须为正数。")
    stride = round(args.mpc_period / dt)
    if not np.isclose(stride * dt, args.mpc_period, rtol=0, atol=1e-10):
        parser.error("MPC 周期必须为 1 ms 的整数倍。")
    controller = MPCController(str(ROOT / "urdf" / "ROKAE_SR4.urdf"),
                               timestep=stride * dt, horizon=args.horizon)
    horizon = controller.horizon
    state_ref, ddq_ref, steps = load_reference(dt, (horizon + 1) * stride)
    if args.duration is not None:
        steps = min(steps, int(np.ceil(args.duration / dt)))

    model = mujoco.MjModel.from_xml_path(str(ROOT / "xml" / "ROKAE_SR4.XML"))
    model.opt.gravity[:] = [0, 0, -9.81]  # 与 Pinocchio 控制模型一致。
    model.opt.timestep = dt  # 每 1 ms 写入力矩；没有新结果时重复写入旧力矩。
    if not args.contacts:
        # 原 XML 的底座/第一连杆网格重叠；先评估自由空间动力学跟踪。
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    if (model.nq, model.nv, model.nu) != (6, 6, 6):
        raise ValueError("当前控制器只支持 6 关节、6 力矩执行器模型。")

    data = mujoco.MjData(model)
    reference_data = mujoco.MjData(model)
    data.qpos[:] = state_ref[0, :6]  # 从轨迹起点开始，避免混入初始定位误差。
    data.qvel[:] = state_ref[0, 6:]
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    if site_id < 0:
        raise ValueError("MuJoCo 模型中缺少 tool_site。")

    # 启动计时前准备首个力矩，避免从零力矩开始等待一次求解。
    initial_start = time.perf_counter()
    tau = controller.compute_control(
        state_ref[0], state_ref[:horizon*stride+1:stride], ddq_ref[:horizon*stride:stride])
    initial_solve_ms = 1000 * (time.perf_counter() - initial_start)
    rows, updates = [], []
    summary = {"status": "completed", "control_dt_s": dt, "horizon": horizon,
               "mpc_period_s": controller.timestep, "initial_solve_ms": initial_solve_ms,
               "mpc_deadline_misses": 0,
               "physics_dt_s": model.opt.timestep, "gravity": model.opt.gravity.tolist(),
               "contacts_enabled": args.contacts, "requested_steps": steps,
               "qp_failures": 0, "trajectory": str(TRAJECTORY),
               "timing_note": "1 kHz ZOH; asynchronous MPC targets next slot; late results discarded; soft realtime"}
    window = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)
    print(f"力矩输入：{1/dt:g} Hz，MPC：{1/controller.timestep:g} Hz，"
          f"预测时长：{horizon*controller.timestep:g} s，零阶保持")
    pending, pending_tick = None, 0
    with window as viewer, ThreadPoolExecutor(max_workers=1) as worker:
        wall_start = time.perf_counter()
        next_render = wall_start
        for k in range(steps):
            if viewer is not None and not viewer.is_running():
                summary["status"] = "window_closed"
                break
            # 以绝对时钟安排 1 ms 步进，后台求解期间主循环照常运行。
            deadline = wall_start + k * dt
            time.sleep(max(0.0, deadline - time.perf_counter()))
            cycle_start = time.perf_counter()
            wall_lag_ms = 1000 * max(0.0, cycle_start - deadline)
            solve_ms, updated = 0.0, int(k == 0)

            if k % stride == 0:
                if pending is not None and pending.done():
                    try:
                        new_tau, solve_ms, finished = pending.result()
                    except RuntimeError as error:
                        summary.update(status="qp_failed", qp_failures=1, error=str(error))
                        break
                    # 只在目标时刻使用按时完成的结果，不能把迟到结果追溯应用。
                    accepted = pending_tick == k and finished <= wall_start + pending_tick * dt
                    updates.append({"request_time_s": (pending_tick-stride)*dt,
                                    "target_time_s": pending_tick*dt,
                                    "finished_wall_s": finished-wall_start,
                                    "solve_ms": solve_ms, "applied": int(accepted)})
                    if accepted:
                        tau, updated = new_tau, 1
                    pending = None
                if k > 0 and not updated:
                    summary["mpc_deadline_misses"] += 1

                # 一个后台任务最多在途一次，不累积过期状态。
                if pending is None and k + stride < steps:
                    pending_tick = k + stride
                    stop = pending_tick + horizon * stride
                    pending = worker.submit(
                        solve_next, controller, np.r_[data.qpos, data.qvel], tau.copy(),
                        state_ref[pending_tick:stop+1:stride], ddq_ref[pending_tick:stop:stride])

            # 这里每 1 ms 都执行；MPC 忙时 tau 保持不变，就是零阶保持。
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)  # 更新积分后的位置对应的 tool_site。
            if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                    or abs(data.time - (k+1)*dt) > 1e-6):
                summary["status"] = "simulation_failed"
                break

            # 比较同一时刻：实际 x[k+1] 对应参考 x_ref[k+1]。
            target = state_ref[k+1]
            reference_data.qpos[:] = target[:6]
            mujoco.mj_kinematics(model, reference_data)
            cycle_ms = 1000 * (time.perf_counter() - cycle_start)
            rows.append(np.r_[data.time, target[:6], data.qpos, target[6:], data.qvel,
                              tau, solve_ms, cycle_ms,
                              reference_data.site_xpos[site_id], data.site_xpos[site_id],
                              updated, wall_lag_ms])

            if (k+1) % round(1/dt) == 0:
                print(f"t={data.time:.1f} s，最大关节误差="
                      f"{np.max(np.abs(np.rad2deg(data.qpos-target[:6]))):.4f} deg")
            # 图形界面只需约 60 Hz，力矩输入仍是 1000 Hz。
            if viewer is not None and time.perf_counter() >= next_render:
                viewer.sync()
                next_render = time.perf_counter() + 1/60
        summary["wall_elapsed_s"] = time.perf_counter() - wall_start

    # 关闭窗口/结束仿真时，收集仍在途的任务，但不再施加其力矩。
    if pending is not None:
        try:
            _, solve_ms, finished = pending.result()
            updates.append({"request_time_s": (pending_tick-stride)*dt,
                            "target_time_s": pending_tick*dt,
                            "finished_wall_s": finished-wall_start,
                            "solve_ms": solve_ms, "applied": 0})
        except RuntimeError as error:
            summary.update(status="qp_failed", qp_failures=1, error=str(error))
    save_results(rows, updates, args.output, summary)
    if summary["status"] in ("qp_failed", "simulation_failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
