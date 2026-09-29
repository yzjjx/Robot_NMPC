"""三个 MuJoCo 示例共用的文件读取、仿真初始化和结果保存函数。"""

from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def period_steps(period, dt):
    """把秒转换成仿真步数，禁止通过四舍五入悄悄改变控制频率。"""
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(period) or period < dt:
        raise ValueError("周期必须为正数，且控制周期不能小于仿真步长。")
    count = int(round(period / dt))
    if not np.isclose(period / dt, count, rtol=0, atol=1e-8):
        raise ValueError("控制周期必须是 XML 仿真步长的整数倍。")
    return count


def load_reference(path, dt, padding_steps):
    """输入：时间、6 个 q、6 个 dq；输出：状态、加速度、仿真步数。

    单位分别是 s、rad、rad/s；加速度由速度差分得到，单位 rad/s²。
    按文件中的真实时间插值，文件采样周期可以与 XML 的步长不同。
    """
    raw = np.loadtxt(path, skiprows=1)
    if raw.ndim != 2 or raw.shape[1] != 13 or len(raw) < 3:
        raise ValueError("轨迹需要至少 3 行、13 列：时间、6 个位置、6 个速度。")
    if not np.isfinite(raw).all() or np.any(np.diff(raw[:, 0]) <= 0):
        raise ValueError("轨迹必须全部有限，时间必须严格递增。")
    source_time = raw[:, 0] - raw[0, 0]
    source_ddq = np.gradient(raw[:, 7:13], source_time, axis=0, edge_order=2)
    steps = int(np.ceil(source_time[-1] / dt))
    times = np.arange(steps + 1) * dt
    q = np.column_stack([np.interp(times, source_time, raw[:, 1+j])
                         for j in range(6)])
    dq = np.column_stack([np.interp(times, source_time, raw[:, 7+j], right=0.0)
                          for j in range(6)])
    ddq = np.column_stack([np.interp(times, source_time, source_ddq[:, j], right=0.0)
                           for j in range(6)])
    # 预测窗口超过文件末尾时，保持最后的位置，速度和加速度补零。
    padding = ((0, padding_steps), (0, 0))
    return (np.column_stack((np.pad(q, padding, mode="edge"),
                             np.pad(dq, padding))),
            np.pad(ddq, padding), steps)


def create_simulation(contacts):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "xml" / "ROKAE_SR4.XML"))
    if (model.nq, model.nv, model.nu) != (6, 6, 6):
        raise ValueError("示例需要 6 个关节和 6 个力矩执行器。")
    # 附加力矩控制：模拟机器人底层已补偿重力，控制器也必须关闭重力项。
    model.opt.gravity[:] = 0
    if not contacts:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return model, mujoco.MjData(model)


def limit_steps(steps, duration, dt):
    if duration is not None:
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("DURATION 必须为正数或 None。")
        steps = min(steps, int(np.floor(duration / dt + 1e-9)))
    if steps < 1:
        raise ValueError("仿真时间至少需要一个 XML 时间步。")
    return steps


def check_torque(torque):
    torque = np.asarray(torque, dtype=float)
    if torque.shape != (6,) or not np.isfinite(torque).all():
        raise ValueError("控制器输出必须是 6 个有限力矩，单位 N·m。")
    return torque.copy()


def advance(model, data, tau, step, dt):
    """零阶保持：两个控制时刻之间，一直施加同一个力矩。"""
    data.ctrl[:] = tau
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
            or abs(data.time - (step + 1) * dt) > 1e-6):
        raise RuntimeError("MuJoCo 状态无效或仿真时间发生重置。")


def write_csv(path, rows, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    # 即使中途失败，也用空表覆盖上次结果，避免误读旧文件。
    values = np.asarray(rows, dtype=float).reshape(-1, len(names))
    np.savetxt(path, values, delimiter=",", header=",".join(names), comments="")


def save_results(rows, output_dir, elapsed, status):
    # 每行对应积分后的状态，以及刚才一个积分步实际使用的力矩。
    names = (["time_s"] + [f"q_rad_{j+1}" for j in range(6)]
             + [f"dq_rad_s_{j+1}" for j in range(6)]
             + [f"tau_Nm_{j+1}" for j in range(6)]
             + [f"q_ref_rad_{j+1}" for j in range(6)])
    write_csv(output_dir / "tracking.csv", rows, names)
    if rows:
        values = np.asarray(rows)
        error = values[:, 1:7] - values[:, 19:25]
        print("各关节位置 RMSE [deg]：", np.rad2deg(np.sqrt(np.mean(error**2, axis=0))))
        print(f"仿真 / 实际运行时间：{values[-1, 0]:.3f} / {elapsed:.3f} s")
    (output_dir / "status.txt").write_text(status + "\n", encoding="utf-8")
    print(f"结果：{output_dir.resolve()}，状态：{status}")
