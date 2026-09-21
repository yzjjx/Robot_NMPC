"""在 MuJoCo 界面中对 ROKAE SR4 进行零力矩仿真。"""

import time
# path用来处理文件路径
from pathlib import Path

import mujoco
# 导入mujoco图形界面
import mujoco.viewer


# 当前文件位于 Robot_NMPC/python_code，模型位于 Robot_NMPC/xml。
# __file__表示代表当前python文件，即python_code\zero_torque_sim.py，Path()函数表示将当前字符串路径转换为Path对象，然后.resolve()表示将其转换为绝对路径
# .parent表示上一级目录，第一个上一级道道python_code，第二个.parent到达python_code的上一级
MODEL_PATH = Path(__file__).resolve().parent.parent / "xml" / "ROKAE_SR4.XML"

# 定义主函数，-> None表示这个函数没有返回值，属于类型标注，没有特定意义，也可以删掉
def main() -> None:
    # 加载机器人模型，并创建保存仿真状态的数据对象
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    # 保存机器人当前时刻的状态
    data = mujoco.MjData(model)

    # launch_passive 只负责显示，仿真步进由下面的循环控制
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 只要mujoco窗口没关闭，就一直进行仿真
        while viewer.is_running():
            # 记录当前真实时间
            step_start = time.time()

            # 六个电机的输入力矩始终为 0 N·m。
            data.ctrl[:] = 0.0
            # 根据当前状态和控制输入，计算下一时刻的机器人状态，这个地方与xml文件里面的timestep有关，目前timestep为0.02s也就是20ms，也就是调用一次下面的语句，仿真时间前进0.02s
            mujoco.mj_step(model, data)

            # 将最新仿真状态同步到显示窗口
            viewer.sync()

            # 按模型的 timestep 实时播放，避免仿真运行得过快
            # 假如上面一段代码执行时间为3ms，就要sleep（0.02-0.003）s
            # 计算当前循环用了多久
            elapsed = time.time() - step_start
            time.sleep(max(0.0, model.opt.timestep - elapsed))

# 如果当前python文件是直接运行的，就执行main函数
if __name__ == "__main__":
    main()
