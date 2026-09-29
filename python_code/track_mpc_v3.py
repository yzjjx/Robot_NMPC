"""v3：后台准备下一块模型，主线程使用当前模型计算反馈力矩。"""

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
REALTIME = False         # 数值模式允许等待；True 时按墙钟运行，超时停止。
MODEL_PERIOD = 0.01      # 模型更新周期：10 ms。
FEEDBACK_RATIO = 2      # 每块反馈 2 次，即反馈周期 5 ms。
# C++ 默认 ratio=3，但 10/3 ms 不是 XML 的 1 ms 步长的整数倍，这里用 2。
HORIZON = 25
TORQUE_LIMIT = 10.0
INTEGRATION_STEP = 0.001  # 控制器内部积分步长，与 MuJoCo 步长分别设置。
OUTPUT_DIR = ROOT / "data_out" / "mpc_v3"


def make_config(module, model_period, ratio, horizon, torque_limit, integration_step, realtime):
    """v3 与 FPGA 导出相同接口，但 Config 必须由各自的模块创建。"""
    if not hasattr(module, "Config") or not hasattr(module.MPCController, "prepare"):
        raise ImportError("加载的扩展不是当前 prepare/feedback 接口，请检查 Python 模块路径。")
    config = module.Config()
    config.model_dt = model_period
    config.feedback_ratio = ratio
    config.horizon = horizon
    config.torque_limit = torque_limit
    config.integration_step = integration_step
    config.gravity_compensated = True
    config.enforce_feedback_budget = realtime
    return config


def preparation_row(info):
    return [info.block, info.total_ms, info.model_ms, info.matrices_and_qp_ms,
            info.backend.batches, info.backend.samples, info.backend.transfer_and_wait_ms]


def run_simulation(controller, config, *, trajectory, headless, duration, contacts,
                   viewer_hz, realtime, output_dir, backend_check=None):
    """CPU v3 和 FPGA 共用同一仿真循环，仅动力学后端不同。

    prepare 输入：块编号、预测起点、(N+1,12) 状态参考、(N,6) 加速度参考。
    feedback 输入：实际 12 维状态、反馈编号；输出中 torque 是 6 维力矩。
    numerical 模式会暂停等待；墙钟模式仍是软件仿真，不模拟通信和计算延迟对
    连续物理对象的完整影响（同步 feedback 期间 MuJoCo 不推进）。
    """
    model, data = create_simulation(contacts)
    dt = float(model.opt.timestep)
    model_steps = period_steps(config.model_dt, dt)
    feedback_steps = period_steps(config.feedback_dt, dt)
    if not np.isfinite(viewer_hz) or viewer_hz <= 0:
        raise ValueError("VIEWER_HZ 必须为正数。")
    offsets = np.arange(config.horizon + 1) * model_steps
    states, accelerations, steps = load_reference(
        trajectory, dt, int(offsets[-1] + model_steps))
    steps = limit_steps(steps, duration, dt)
    data.qpos[:], data.qvel[:] = states[0, :6], states[0, 6:]
    mujoco.mj_forward(model, data)
    if backend_check is not None:
        # 同时检查预测窗口覆盖的轨迹，不仅检查本次仿真经过的部分。
        backend_check(controller, states)

    first = controller.prepare(0, states[0], states[offsets], accelerations[offsets[:-1]])
    preparation_rows = [preparation_row(first)]  # 启动耗时也保存。
    feedback_rows = []
    rows = []
    status = "completed"
    failure = None
    pending = None
    pending_block = None
    tau = np.zeros(6)
    window = nullcontext(None) if headless else mujoco.viewer.launch_passive(model, data)
    print(f"后端：{controller.backend_name}，模型 {1/config.model_dt:g} Hz，"
          f"反馈 {1/config.feedback_dt:g} Hz，REALTIME={realtime}")
    print("数值模式允许等待计算，其完成情况不能证明达到实时控制频率。")
    with window as viewer, ThreadPoolExecutor(max_workers=1) as worker:
        wall_start = time.perf_counter()
        next_render = wall_start
        try:
            for k in range(steps):
                if viewer is not None and not viewer.is_running():
                    status = "window_closed"
                    break
                target = wall_start + k*dt
                if realtime:
                    time.sleep(max(0.0, target-time.perf_counter()))

                if k % feedback_steps == 0:
                    tick = k // feedback_steps
                    block, phase = divmod(tick, config.feedback_ratio)
                    if phase == 0 and block > 0:
                        if pending is None or pending_block != block:
                            raise RuntimeError("缺少当前块的准备任务。")
                        if realtime and not pending.done():
                            raise RuntimeError("下一块模型准备超时，停止仿真。")
                        job, pending = pending, None
                        info, prepared_at = job.result()
                        preparation_rows.append(preparation_row(info))
                        if realtime and prepared_at > target:
                            raise RuntimeError("当前块模型已过期，停止仿真。")

                    # 状态在主线程复制，准备线程只接收独立的数组。
                    sampled = time.perf_counter()
                    state = np.r_[data.qpos, data.qvel]
                    result = controller.feedback(state, tick, time.perf_counter()-sampled)
                    completed = time.perf_counter()
                    applied = not realtime or completed <= target + config.feedback_dt
                    feedback_rows.append([tick, k*dt, block, phase,
                                          1000*(completed-sampled), result.total_ms,
                                          result.solve_ms, result.qp_iterations,
                                          int(result.deadline_missed), int(applied)])
                    if not applied:
                        raise RuntimeError("反馈求解超过墙钟截止时间，停止仿真。")
                    tau = check_torque(result.torque)

                    # 第一相位反馈后，提前准备下块。states[1] 位于下一模型块起点，
                    # 而不是简单地取“下一个反馈时刻”。二者的时间间隔不同。
                    if phase == 0 and (block+1)*model_steps < steps:
                        pending_block = block + 1
                        indices = pending_block*model_steps + offsets
                        pending = worker.submit(
                            prepare_next, controller, pending_block, result.states[1].copy(),
                            states[indices], accelerations[indices[:-1]])

                advance(model, data, tau, k, dt)
                rows.append(np.r_[data.time, data.qpos, data.qvel, tau, states[k+1, :6]])
                if viewer is not None and time.perf_counter() >= next_render:
                    viewer.sync()
                    next_render = time.perf_counter() + 1.0/viewer_hz
        except Exception as error:
            failure = error
            status = f"failed: {error}"
        finally:
            if pending is not None:
                try:
                    info, _ = pending.result()
                    preparation_rows.append(preparation_row(info))
                except Exception as error:
                    if failure is None:
                        failure = error
                        status = f"failed: {error}"

    save_results(rows, output_dir, time.perf_counter()-wall_start, status)
    write_csv(output_dir / "preparation.csv", preparation_rows,
              ["block", "total_ms", "model_ms", "matrices_qp_ms", "aba_batches",
               "aba_samples", "transfer_wait_ms"])
    write_csv(output_dir / "feedback.csv", feedback_rows,
              ["tick", "time_s", "block", "phase", "wall_ms", "cpp_ms", "qp_ms",
               "qp_iterations", "cpp_deadline_missed", "applied"])
    if failure is not None:
        raise RuntimeError(status) from failure


def prepare_next(controller, block, anchor, states, accelerations):
    """完成时间由工作线程记录，避免主线程晚醒后误认为任务按时完成。"""
    info = controller.prepare(block, anchor, states, accelerations)
    return info, time.perf_counter()


def main():
    import rokae_mpc_v3 as mpc

    config = make_config(mpc, MODEL_PERIOD, FEEDBACK_RATIO, HORIZON,
                         TORQUE_LIMIT, INTEGRATION_STEP, REALTIME)
    controller = mpc.MPCController(str(ROOT / "urdf" / "ROKAE_SR4.urdf"),
                                   config, backend="cpu")
    run_simulation(controller, config, trajectory=TRAJECTORY, headless=HEADLESS,
                   duration=DURATION, contacts=CONTACTS, viewer_hz=VIEWER_HZ,
                   realtime=REALTIME, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
