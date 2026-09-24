"""按 XML 时间步长运行 MuJoCo，并用后台 MPC 和零阶保持跟踪轨迹。"""

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
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

# 仿真配置：需要修改参数时，直接修改这里。
HEADLESS = False       # False：显示 MuJoCo 窗口；True：不显示窗口。
DURATION = None        # None：运行完整轨迹；也可以设置为需要运行的秒数。
CONTACTS = False       # False：关闭接触；True：保留接触作用。
MPC_PERIOD = 0.1      # MPC 控制周期，单位为秒。
HORIZON = 15           # MPC 预测步数。
OUTPUT_DIR = ROOT / "data_out" 

# 将轨迹文件整理成控制器能够使用的数据
# dt为仿真周期，padding_steps为轨迹末尾补充多少个点
def load_reference(dt, padding_steps):
    """读取均匀采样轨迹；采样周期与仿真步长相同时直接使用原始数据。"""
    raw = np.loadtxt(TRAJECTORY, skiprows=1)

    if raw.ndim != 2 or raw.shape[1] != 13 or len(raw) < 3:
        raise ValueError("轨迹文件需要时间、6 个位置、6 个速度，共 13 列。")
    
    if not np.isfinite(raw).all():
        raise ValueError("轨迹数据必须有限。")

    # 输入保证均匀采样，因此只需用前两个时间点确定固定周期。
    trajectory_dt = raw[1, 0] - raw[0, 0]# 计算轨迹周期
    if trajectory_dt <= 0:
        raise ValueError("轨迹采样周期必须为正数。")

    q = raw[:, 1:7]    # 六个关节的位置
    dq = raw[:, 7:13]  # 六个关节的速度
    ddq = np.gradient(dq, trajectory_dt, axis=0, edge_order=2)
    # 假设有1001个状态，中间就是1000个区间
    steps = len(raw) - 1  # 相邻两个状态之间对应一个仿真步

    # 周期相同（当前均为 1 ms）时无需插值；不同时才按仿真时刻重采样
    # 也就是需要轨迹的输入频率为1ms，xml的timestep也是1ms
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
    # 位置默认补edge值，也就是末端最后一个数
    q = np.pad(q, padding, mode="edge")
    # 速度默认补常量0
    dq = np.pad(dq, padding, mode="constant")
    ddq = np.pad(ddq, padding, mode="constant")
    return np.column_stack((q, dq)), ddq, steps

# 在后台线程中，提前计算下一个控制时刻的力矩。
def solve_next(controller, state, held_tau, states, accelerations):
    """先预测旧力矩保持一个周期后的状态，再计算下一时刻的力矩。"""
    start = time.perf_counter()
    predicted_state = controller.predict_state(
        state, held_tau, controller.timestep)
    tau = controller.compute_control(predicted_state, states, accelerations)
    return tau, 1000 * (time.perf_counter() - start)


def save_results(rows, output_dir, wall_elapsed, status):
    """保存实际关节位置和速度。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        result = np.asarray(rows)
        names = (["time_s"] + [f"q_rad_{j+1}" for j in range(6)]
                 + [f"dq_rad_s_{j+1}" for j in range(6)])
        np.savetxt(output_dir / "tracking.csv", result, delimiter=",",
                   header=",".join(names), comments="")
        print(f"仿真/实际运行时间：{result[-1, 0]:.3f} / {wall_elapsed:.3f} s")
    print(f"结果：{output_dir.resolve()}，状态：{status}")


def main():
    if DURATION is not None and (not np.isfinite(DURATION) or DURATION <= 0):
        raise ValueError("DURATION 必须为正数。")

    # 加载mujoco模型
    model = mujoco.MjModel.from_xml_path(str(ROOT / "xml" / "ROKAE_SR4.XML"))
    # 加载时间步长
    dt = float(model.opt.timestep)
    mpc_period = MPC_PERIOD
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("MuJoCo 时间步长必须为正数。")
    if not np.isfinite(mpc_period) or mpc_period < dt or HORIZON < 1:
        raise ValueError("MPC 周期必须至少为仿真步长，预测步数必须为正数。")
    if not np.isclose(mpc_period / dt, round(mpc_period / dt), rtol=0, atol=1e-8):
        raise ValueError("MPC 周期必须为仿真步长的整数倍。")
    controller = MPCController(str(ROOT / "urdf" / "ROKAE_SR4.urdf"),
                               timestep=mpc_period, horizon=HORIZON)
    horizon = controller.horizon
    # MPC参考点相隔0.1秒，例如偏移量为第0、100、200...行。
    reference_offsets = np.rint(np.arange(horizon + 1) * mpc_period / dt).astype(int)
    control_steps = int(round(mpc_period / dt))
    state_ref, ddq_ref, steps = load_reference(
        dt, int(reference_offsets[-1] + control_steps))

    if DURATION is not None:
        steps = min(steps, int(np.ceil(DURATION / dt)))

    model.opt.gravity[:] = [0, 0, 0]  # 模拟底层已补偿重力，与 Pinocchio 控制模型一致。
    if not CONTACTS:
        # 原 XML 的底座/第一连杆网格重叠；先评估自由空间动力学跟踪。
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    if (model.nq, model.nv, model.nu) != (6, 6, 6):
        raise ValueError("当前控制器只支持 6 关节、6 力矩执行器模型。")

    data = mujoco.MjData(model)
    data.qpos[:] = state_ref[0, :6]  # 从轨迹起点开始，避免混入初始定位误差。
    data.qvel[:] = state_ref[0, 6:]
    mujoco.mj_forward(model, data)

    # 启动计时前准备首个力矩，避免从零力矩开始等待一次求解。
    solve_start = time.perf_counter()
    tau = controller.compute_control(
        state_ref[0], state_ref[reference_offsets], ddq_ref[reference_offsets[:-1]])
    print(f"MPC t=0.000 s 耗时：{1000 * (time.perf_counter() - solve_start):.3f} ms")
    rows = []
    status = "completed"
    window = nullcontext(None) if HEADLESS else mujoco.viewer.launch_passive(model, data)
    print(f"力矩输入目标：{1/dt:g} Hz，MPC 周期：{mpc_period:g} s，"
          f"预测时长：{horizon*mpc_period:g} s，下一控制时刻采用，否则保持旧力矩")
    pending = None
    next_control_step = control_steps
    with window as viewer, ThreadPoolExecutor(max_workers=1) as worker:
        wall_start = time.perf_counter()
        next_render = wall_start

        # t=0时开始准备t=0.1秒的力矩。
        if next_control_step < steps:
            pending_target_step = next_control_step
            indices = pending_target_step + reference_offsets
            pending = worker.submit(
                solve_next, controller, np.r_[data.qpos, data.qvel], tau.copy(),
                state_ref[indices], ddq_ref[indices[:-1]])

        for k in range(steps):
            if viewer is not None and not viewer.is_running():
                status = "window_closed"
                break
            # 以绝对时钟安排仿真步进，后台求解期间主循环照常运行。
            step_time = wall_start + k * dt
            time.sleep(max(0.0, step_time - time.perf_counter()))

            if k == next_control_step:
                # 到达控制时刻：结果已完成就使用，否则继续保持旧力矩。
                if pending is not None and pending.done():
                    try:
                        tau, solve_ms = pending.result()
                    except RuntimeError as error:
                        status = "qp_failed"
                        print(f"MPC 求解失败：{error}")
                        pending = None
                        break
                    print(f"MPC t={pending_target_step * dt:.3f} s 耗时：{solve_ms:.3f} ms")
                    pending = None

                next_control_step += control_steps

                # 后台空闲后，提前准备下一个控制时刻的力矩。
                if pending is None and next_control_step < steps:
                    pending_target_step = next_control_step
                    indices = pending_target_step + reference_offsets
                    pending = worker.submit(
                        solve_next, controller, np.r_[data.qpos, data.qvel], tau.copy(),
                        state_ref[indices], ddq_ref[indices[:-1]])

            # 每个 XML 时间步都执行；MPC 忙时 tau 保持不变，就是零阶保持。
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)  # 更新积分后的位置对应的 tool_site。
            if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                    or abs(data.time - (k+1)*dt) > 1e-6):
                status = "simulation_failed"
                break

            rows.append(np.r_[data.time, data.qpos, data.qvel])

            if (k+1) % round(1/dt) == 0:
                target = state_ref[k+1]
                print(f"t={data.time:.1f} s，最大关节误差="
                      f"{np.max(np.abs(np.rad2deg(data.qpos-target[:6]))):.4f} deg")
            # 仿真仍每 1 ms 前进一步；这里只把最新状态约每秒显示 60 次。
            if viewer is not None and time.perf_counter() >= next_render:
                viewer.sync()
                next_render = time.perf_counter() + 1.0 / VIEWER_HZ
        wall_elapsed = time.perf_counter() - wall_start

    # 关闭窗口/结束仿真时，收集仍在途的任务，但不再施加其力矩。
    if pending is not None:
        try:
            _, solve_ms = pending.result()
            print(f"MPC t={pending_target_step * dt:.3f} s 耗时：{solve_ms:.3f} ms")
        except RuntimeError as error:
            status = "qp_failed"
            print(f"MPC 求解失败：{error}")
    save_results(rows, OUTPUT_DIR, wall_elapsed, status)
    if status in ("qp_failed", "simulation_failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
