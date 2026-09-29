"""v2：MuJoCo 逐步仿真，后台提前计算下一控制时刻的力矩。"""

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import time

import mujoco
import mujoco.viewer
import numpy as np

from mpc_sim_utils import (ROOT, advance, check_torque, create_simulation,
                           limit_steps, load_reference, period_steps,
                           save_results, write_csv)

TRAJECTORY = ROOT / "data_in" / "circle_R200_joint_trajectory_SR4_V50.txt"
HEADLESS = False
DURATION = None
CONTACTS = False
VIEWER_HZ = 60.0
REALTIME = False       # False：数值仿真，允许等待求解；True：按墙钟运行并检查超时。
MPC_PERIOD = 0.01     # 0.01 s = 100 Hz；改成 0.02 s 即为 50 Hz。
HORIZON = 25
TORQUE_LIMIT = 10.0   # N·m；三个示例采用相同限幅，便于比较。
OUTPUT_DIR = ROOT / "data_out" / "mpc_v2"


def solve_next(controller, state, held_tau, states, accelerations):
    """输入当前状态和旧力矩，输出下一时刻力矩及耗时统计。

    state 前 6 项是 q，后 6 项是 dq。先预测旧力矩保持一个周期后的
    状态，再求解该时刻的控制量；求出的力矩在下个控制时刻才使用。
    """
    start = time.perf_counter()
    predicted = controller.predict_state(state, held_tau, controller.timestep)
    tau = check_torque(controller.compute_control(predicted, states, accelerations))
    timing = controller.timing  # 在同一个工作线程读取，避免与求解争用锁。
    completed = time.perf_counter()
    return tau, completed, [1000*(completed-start), timing.total_ms,
                            timing.qp_ms, timing.qp_iterations]


def main():
    from rokae_mpc_v2 import MPCController

    model, data = create_simulation(CONTACTS)
    dt = float(model.opt.timestep)
    control_steps = period_steps(MPC_PERIOD, dt)
    if HORIZON < 1 or VIEWER_HZ <= 0 or not np.isfinite(TORQUE_LIMIT) or TORQUE_LIMIT <= 0:
        raise ValueError("HORIZON、VIEWER_HZ、TORQUE_LIMIT 必须为正数。")
    offsets = np.arange(HORIZON + 1) * control_steps
    states, accelerations, steps = load_reference(
        TRAJECTORY, dt, int(offsets[-1] + control_steps))
    steps = limit_steps(steps, DURATION, dt)
    controller = MPCController(str(ROOT / "urdf" / "ROKAE_SR4.urdf"),
                               timestep=MPC_PERIOD, horizon=HORIZON,
                               gravity_compensated=True, integration_step=dt)
    controller.set_torque_limits(-np.ones(6)*TORQUE_LIMIT, np.ones(6)*TORQUE_LIMIT)
    data.qpos[:], data.qvel[:] = states[0, :6], states[0, 6:]
    mujoco.mj_forward(model, data)

    # 冷启动不占用第一个控制周期，但单独记录其耗时。
    start = time.perf_counter()
    tau = check_torque(controller.compute_control(
        states[0], states[offsets], accelerations[offsets[:-1]]))
    initial_ms = 1000*(time.perf_counter()-start)
    timing = controller.timing
    solve_rows = [[0, initial_ms, timing.total_ms, timing.qp_ms, timing.qp_iterations, 1]]
    rows = []
    failure = None
    status = "completed"
    pending = None
    target_step = control_steps
    window = nullcontext(None) if HEADLESS else mujoco.viewer.launch_passive(model, data)
    print(f"v2：仿真步长 {dt:g} s，MPC {1/MPC_PERIOD:g} Hz，REALTIME={REALTIME}")
    print("非实时模式允许暂停仿真等待计算，不能据此判断控制器是否达到目标频率。")
    with window as viewer, ThreadPoolExecutor(max_workers=1) as worker:
        wall_start = time.perf_counter()
        next_render = wall_start
        try:
            for k in range(steps):
                if viewer is not None and not viewer.is_running():
                    status = "window_closed"
                    break
                if REALTIME:
                    time.sleep(max(0.0, wall_start + k*dt - time.perf_counter()))

                if k == target_step:
                    if REALTIME and not pending.done():
                        raise RuntimeError("v2 求解未赶上下一个控制时刻。")
                    job, pending = pending, None
                    new_tau, completed, timing_row = job.result()
                    applied = not REALTIME or completed <= wall_start + k*dt
                    solve_rows.append([k*dt, *timing_row, int(applied)])
                    if not applied:
                        raise RuntimeError("v2 结果已过期，停止仿真。")
                    tau = new_tau
                    target_step += control_steps

                # 每个控制时刻只提交一个任务；工作线程不访问 MuJoCo 的可变状态。
                if k % control_steps == 0 and target_step < steps:
                    indices = target_step + offsets
                    pending = worker.submit(
                        solve_next, controller, np.r_[data.qpos, data.qvel], tau.copy(),
                        states[indices], accelerations[indices[:-1]])

                advance(model, data, tau, k, dt)
                rows.append(np.r_[data.time, data.qpos, data.qvel, tau, states[k+1, :6]])
                if viewer is not None and time.perf_counter() >= next_render:
                    viewer.sync()
                    next_render = time.perf_counter() + 1.0/VIEWER_HZ
        except Exception as error:
            failure = error
            status = f"failed: {error}"
        finally:
            # 退出前收集后台任务；窗口关闭或超时时，不施加其输出。
            if pending is not None:
                try:
                    _, _, timing_row = pending.result()
                    solve_rows.append([target_step*dt, *timing_row, 0])
                except Exception as error:
                    if failure is None:
                        failure = error
                        status = f"failed: {error}"

    save_results(rows, OUTPUT_DIR, time.perf_counter()-wall_start, status)
    write_csv(OUTPUT_DIR / "solves.csv", solve_rows,
              ["target_time_s", "predict_and_solve_ms", "cpp_solve_ms", "qp_ms",
               "qp_iterations", "applied"])
    if failure is not None:
        raise RuntimeError(status) from failure


if __name__ == "__main__":
    main()
