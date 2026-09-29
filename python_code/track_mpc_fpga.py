"""FPGA 控制器 + MuJoCo；默认 mock，不打开 FPGA 设备，也不连接机器人。

复用 v3 的 prepare/feedback 仿真循环：FPGA 参与准备阶段的 ABA 计算，
当前实现的解析导数和 QP 仍在 CPU 上计算。mock 耗时不代表实际 FPGA 耗时。
"""

import numpy as np

from mpc_sim_utils import ROOT
from track_mpc_v3 import make_config, run_simulation

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
    import rokae_mpc_fpga as mpc

    if BACKEND not in ("cpu", "mock", "fpga"):
        raise ValueError("BACKEND 只能是 cpu、mock 或 fpga。")
    if BACKEND == "fpga" and not ALLOW_HARDWARE:
        raise ValueError("使用真实 FPGA 前，需要设置 ALLOW_HARDWARE=True。")
    if not np.isfinite(TORQUE_LIMIT) or not 0 < TORQUE_LIMIT <= 10:
        raise ValueError("FPGA 示例要求 0 < TORQUE_LIMIT <= 10 N·m。")
    config = make_config(mpc, MODEL_PERIOD, FEEDBACK_RATIO, HORIZON,
                         TORQUE_LIMIT, INTEGRATION_STEP, REALTIME)
    controller = mpc.MPCController(str(ROOT / "urdf" / "ROKAE_SR4.urdf"), config,
                                   backend=BACKEND, allow_hardware=ALLOW_HARDWARE,
                                   h2c=H2C, c2h=C2H)
    print("mock 是软件后端，不能用其耗时评价真实 FPGA 加速效果。")
    run_simulation(controller, config, trajectory=TRAJECTORY, headless=HEADLESS,
                   duration=DURATION, contacts=CONTACTS, viewer_hz=VIEWER_HZ,
                   realtime=REALTIME, output_dir=OUTPUT_DIR,
                   backend_check=validate_backend if BACKEND != "cpu" else None)


if __name__ == "__main__":
    main()
