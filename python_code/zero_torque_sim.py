"""在 MuJoCo 界面中对 ROKAE SR4 进行零力矩仿真。"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer


# 当前文件位于 Robot_NMPC/python_code，模型位于 Robot_NMPC/xml。
MODEL_PATH = Path(__file__).resolve().parent.parent / "xml" / "ROKAE_SR4.XML"


def main() -> None:
    # 加载机器人模型，并创建保存仿真状态的数据对象。
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # launch_passive 只负责显示，仿真步进由下面的循环控制。
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 六个电机的输入力矩始终为 0 N·m。
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)

            # 将最新仿真状态同步到显示窗口。
            viewer.sync()

            # 按模型的 timestep 实时播放，避免仿真运行得过快。
            elapsed = time.time() - step_start
            time.sleep(max(0.0, model.opt.timestep - elapsed))


if __name__ == "__main__":
    main()
