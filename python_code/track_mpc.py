"""按 XML 时间步长运行 MuJoCo，并用后台 MPC 和零阶保持跟踪轨迹。"""

import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
# 保存计算结果
import csv
# 处理文件路径
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from rokae_mpc import MPCController

# 输入路径
ROOT = Path(__file__).resolve().parent.parent
TRAJECTORY = ROOT / "data_in" / "circle_R200_joint_trajectory_SR4_V50.txt"
VIEWER_HZ = 60.0  # 只控制画面刷新频率，不改变仿真和力矩输入频率。

# 将轨迹文件整理成控制器能够使用的数据
def load_reference(dt, padding_steps):
    """读取均匀采样轨迹；采样周期与仿真步长相同时直接使用原始数据。"""
    raw = np.loadtxt(TRAJECTORY, skiprows=1)

    if raw.ndim != 2 or raw.shape[1] != 13 or len(raw) < 3:
        raise ValueError("轨迹文件需要时间、6 个位置、6 个速度，共 13 列。")
    
    if not np.isfinite(raw).all():
        raise ValueError("轨迹数据必须有限。")

    # 输入保证均匀采样，因此只需用前两个时间点确定固定周期。
    trajectory_dt = raw[1, 0] - raw[0, 0]
    if trajectory_dt <= 0:
        raise ValueError("轨迹采样周期必须为正数。")

    q = raw[:, 1:7]    # 六个关节的位置
    dq = raw[:, 7:13]  # 六个关节的速度
    ddq = np.gradient(dq, trajectory_dt, axis=0, edge_order=2)
    steps = len(raw) - 1  # 相邻两个状态之间对应一个仿真步

    # 周期相同（当前均为 1 ms）时无需插值；不同时才按仿真时刻重采样
    # 仿真周期为dt，原始轨迹周期为trajectory_dt
    # np.isclose用于判断两个浮点数是否相近，因此下面的代码基本用不到
    # if not np.isclose(dt, trajectory_dt, rtol=1e-9, atol=1e-12):
    #     source_time = np.arange(len(raw)) * trajectory_dt
    #     steps = int(np.ceil(source_time[-1] / dt))
    #     sample_time = np.arange(steps + 1) * dt
    #     q = np.column_stack([
    #         np.interp(sample_time, source_time, q[:, j]) for j in range(6)
    #     ])
    #     dq = np.column_stack([
    #         np.interp(sample_time, source_time, dq[:, j], right=0.0) for j in range(6)
    #     ])
    #     ddq = np.column_stack([
    #         np.interp(sample_time, source_time, ddq[:, j], right=0.0) for j in range(6)
    #     ])

    # 为末尾的 MPC 预测补点：位置保持末点，速度和加速度补零，也就是如果到末尾输入轨迹文件csv已经没有了，就自动补齐
    # 末尾用边界值，也就是最后一个值不断重复
    padding = ((0, padding_steps), (0, 0))  # 只在末尾补行，不增加列
    q = np.pad(q, padding, mode="edge")
    dq = np.pad(dq, padding, mode="constant")
    ddq = np.pad(ddq, padding, mode="constant")
    return np.column_stack((q, dq)), ddq, steps

# 为下一个时刻准备力矩
def solve_next(controller, state, held_tau, states, accelerations):
    """后台任务：预测旧力矩保持一周期后的状态，计算下一更新时刻的力矩。"""
    start = time.perf_counter()
    predicted = controller.predict_state(state, held_tau, controller.timestep)
    tau = controller.compute_control(predicted, states, accelerations)
    finished = time.perf_counter()
    return tau, 1000 * (finished - start), finished


def save_results(rows, updates, output_dir, wall_elapsed, deadline_misses, status):
    """保存 CSV 数据，并在终端显示跟踪误差。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    if updates:
        with (output_dir / "mpc_updates.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(updates[0]))
            writer.writeheader()
            writer.writerows(updates)
        solve_times = np.array([item["solve_ms"] for item in updates])
        print(f"后台 MPC 平均/最大耗时：{np.mean(solve_times):.3f} / "
              f"{np.max(solve_times):.3f} ms")
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
        joint_rmse = np.sqrt(np.mean(error_deg**2, axis=0))
        tcp_rmse = np.sqrt(np.mean(tcp_error_mm**2))
        print("各关节位置 RMSE (deg)：", np.round(joint_rmse, 6))
        print(f"tool_site 位置 RMSE：{tcp_rmse:.4f} mm")
        print(f"仿真/实际运行时间：{result[-1, 0]:.3f} / {wall_elapsed:.3f} s，"
              f"MPC 更新错过次数：{deadline_misses}")
    print(f"结果：{output_dir.resolve()}，状态：{status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="不打开 MuJoCo 窗口")
    parser.add_argument("--duration", type=float, help="只运行前多少秒，默认完整轨迹")
    parser.add_argument("--contacts", action="store_true", help="保留 XML 网格的接触作用")
    parser.add_argument("--mpc-period", type=float, default=0.1, help="MPC 更新周期，默认 0.1 秒")
    parser.add_argument("--horizon", type=int, default=10, help="MPC 预测步数，默认 10")
    parser.add_argument("--output", type=Path, default=ROOT / "data_out" / "mpc_zoh")
    args = parser.parse_args()
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration 必须为正数。")

    model = mujoco.MjModel.from_xml_path(str(ROOT / "xml" / "ROKAE_SR4.XML"))
    dt = float(model.opt.timestep)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("XML 中的 timestep 必须为正数。")
    if not np.isfinite(args.mpc_period) or args.mpc_period < dt or args.horizon < 1:
        parser.error(f"MPC 周期必须至少为 XML 时间步长 {1000*dt:g} ms，预测步数必须为正数。")
    stride = round(args.mpc_period / dt)
    if not np.isclose(stride * dt, args.mpc_period, rtol=0, atol=1e-10):
        parser.error(f"MPC 周期必须为 XML 时间步长 {1000*dt:g} ms 的整数倍。")
    controller = MPCController(str(ROOT / "urdf" / "ROKAE_SR4.urdf"),
                               timestep=stride * dt, horizon=args.horizon)
    horizon = controller.horizon
    state_ref, ddq_ref, steps = load_reference(dt, (horizon + 1) * stride)
    if args.duration is not None:
        steps = min(steps, int(np.ceil(args.duration / dt)))

    model.opt.gravity[:] = [0, 0, -9.81]  # 与 Pinocchio 控制模型一致。
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
    tau = controller.compute_control(
        state_ref[0], state_ref[:horizon*stride+1:stride], ddq_ref[:horizon*stride:stride])
    rows, updates = [], []
    status = "completed"
    deadline_misses = 0
    window = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)
    print(f"力矩输入：{1/dt:g} Hz，MPC：{1/controller.timestep:g} Hz，"
          f"预测时长：{horizon*controller.timestep:g} s，零阶保持")
    pending, pending_tick = None, 0
    with window as viewer, ThreadPoolExecutor(max_workers=1) as worker:
        wall_start = time.perf_counter()
        next_render = wall_start
        for k in range(steps):
            if viewer is not None and not viewer.is_running():
                status = "window_closed"
                break
            # 以绝对时钟安排仿真步进，后台求解期间主循环照常运行。
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
                        status = "qp_failed"
                        print(f"MPC 求解失败：{error}")
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

            # 每个 XML 时间步都执行；MPC 忙时 tau 保持不变，就是零阶保持。
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
            # 仿真仍每 1 ms 前进一步；这里只把最新状态约每秒显示 60 次。
            if viewer is not None and time.perf_counter() >= next_render:
                viewer.sync()
                next_render = time.perf_counter() + 1.0 / VIEWER_HZ
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
