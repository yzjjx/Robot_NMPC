# 1000 Hz 力矩输入、后台 MPC 与零阶保持

`track_mpc.py` 读取 `data_in/circle_R200_joint_trajectory_SR4_V50.txt`，
通过 `rokae_mpc` 模块调用 C++ 控制器，再把力矩输入 MuJoCo。
`zero_torque_sim.py` 是独立的零力矩示例。

`track_mpc.py` 的默认运行方式：

| 部分 | 周期 | 工作内容 |
| --- | --- | --- |
| MuJoCo/力矩输入 | 1 ms，1000 Hz | 每步写入力矩；没有新结果时保持旧力矩 |
| 后台 MPC | 100 ms，10 Hz | 计算下一个更新时刻使用的力矩 |
| 图形界面 | 约 60 Hz | 显示机器人最新状态 |

这是实时推进的闭环仿真，不是计算完成后的动画回放。
后台 C++ 求解释放 Python GIL，因此主循环可以继续步进。
Python/操作系统调度仍会有抖动，这是软实时仿真，不是实机硬实时保证。

## 编译和运行

在工作区根目录（包含 `Robot_NMPC` 的目录）执行：

```bash
cmake -S Robot_NMPC -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build build -j2
python3 Robot_NMPC/python_code/track_mpc.py
```

本机已安装 Python 依赖 `pybind11`、`numpy`、`mujoco`、`matplotlib`。
编译和运行必须使用相同的 Python 环境。CMake 会把编译好的
`rokae_mpc*.so` 放在 `python_code`，脚本可以直接导入。
修改 C++ 后需要重新编译，修改 Python 后直接运行即可。
原 C++ 可执行程序仍然可以构建；只需要 C++ 时设置 `-DBUILD_PYTHON_BINDINGS=OFF`。

不显示窗口，运行完整轨迹并生成报告：

```bash
python3 Robot_NMPC/python_code/track_mpc.py --headless
```

无界面模式也按真实时间运行，以保留计算延迟的影响。
完整轨迹约运行 26.13 秒，另需初始求解、模型加载和报告绘图时间。

只运行前 2 秒，或者开启原 XML 的接触：

```bash
python3 Robot_NMPC/python_code/track_mpc.py --duration 2
python3 Robot_NMPC/python_code/track_mpc.py --contacts --output Robot_NMPC/data_out/mpc_contacts
```

修改慢速 MPC 周期与预测步数，例如每 50 ms 更新一次、预测 20 步：

```bash
python3 Robot_NMPC/python_code/track_mpc.py --mpc-period 0.05 --horizon 20
```

MPC 周期必须为 1 ms 的整数倍，力矩输入始终保持 1000 Hz。
关闭窗口会保存已经完成的数据。默认输出目录为 `data_out/mpc_zoh_1000hz`，
重复运行会更新该目录下的报告；用 `--output` 可以保留不同实验。

## 控制接口

绑定代码是 `src/python_bindings.cpp`。在 `python_code` 目录启动 Python 后可调用：

```python
import numpy as np
from rokae_mpc import MPCController

controller = MPCController("../urdf/ROKAE_SR4.urdf", timestep=0.1, horizon=10)
N = controller.horizon       # 10 段力矩
dt = controller.timestep    # 每段力矩保持 0.1 s，总预测时长为 1 s

state = np.zeros(12)         # 当前状态：[q1...q6, dq1...dq6]
state_ref = np.zeros((N + 1, 12))  # 当前及未来 N 步参考状态
ddq_ref = np.zeros((N, 6))        # 当前及未来 N-1 步参考加速度
tau = controller.compute_control(state, state_ref, ddq_ref)
```

返回的 `tau` 是六维 NumPy 数组，单位 N·m。位置为 rad，速度为 rad/s，
加速度为 rad/s²。一个实验应一直复用同一个 `controller`，以保留上一周期的
优化力矩序列；重新开始实验时创建新实例。输入尺寸错误抛出 `ValueError`，
QP 求解失败抛出 `RuntimeError`，不会把回退力矩当作成功优化。

仿真循环的核心只有：

```python
# 后台任务独立调用 compute_control；主循环不等待它。
# 在更新时刻，只接收按时完成的新力矩，否则继续保持旧力矩。
data.ctrl[:] = tau     # 每 1 ms 都写入，即使 tau 与上一步相同
mujoco.mj_step(model, data)
```

首次启动先计算初始力矩，然后开始仿真时钟。每个 MPC 更新时刻：

1. 接收上个时刻提交的任务；只使用按时完成、目标时间匹配的结果。
2. 如果任务超时，继续保持旧力矩，并记录错过一次更新。迟到结果不会补发。
3. 如果后台空闲，复制当前状态和正在施加的力矩，提交下一个更新时刻的任务。

后台先调用 `predict_state(state, tau, 0.1)`，预测旧力矩保持 100 ms 后的状态，
再使用从那个时刻开始的参考轨迹计算 MPC。这补偿了一周期的计划执行延迟。
任意时刻最多一个任务在途；任务失败会停止实验并保存失败状态。

## 实验设置与结果含义

- 轨迹原始采样间隔为 1 ms，按控制器的 1 ms 周期取参考状态（1000 Hz）。
  参考加速度由文件中的速度对时间求导得到。
- 1000 Hz 输入周期由脚本的 `INPUT_DT = 0.001` 设置；
  MPC 周期由 `--mpc-period` 设置。两者现在是独立参数。
- MPC 默认预测 10 段，每段力矩保持 100 ms，预测时长 1 秒。
  名义预测和有限差分都使用 `compute_held_state()`，以不超过 1 ms 的步长
  积分整段保持时间。因此预测中的力矩保持方式与正常执行时一致。
  发生超时、连续保持旧力矩时，实际执行会偏离原计划，并体现在误差报告中。
- MuJoCo 从参考轨迹第一点的位置和速度开始；这个实验不包含初始定位过程。
- 轨迹结束后，预测窗口的位置保持末点，速度与加速度设为零。
  原文件长度为 26.133 s，完整实验包含 26133 个控制周期。
- 脚本将 MuJoCo 重力设为 `(0, 0, -9.81)`，与 Pinocchio 默认值一致；
  物理仿真步长为 1 ms，每个物理步之前写入当前保持的力矩。
  这些设置只修改内存中的模型。
- 原 XML 在轨迹起点存在底座与第一连杆的网格重叠接触。
  默认关闭接触，评估自由空间跟踪；`--contacts` 可保留原模型的接触。
  因此默认结果不代表碰撞或接触任务中的性能。
- 误差在同一时刻计算：执行第 k 个力矩后，用实际 `x[k+1]` 对比 `x_ref[k+1]`。
- Q、R 和力矩限制沿用原实现。预测步数和控制保持周期已按上述多频率方案设置。
  C++ 构造函数以及 Python 构造函数仍保留 `timestep=0.001, horizon=40`
  的默认参数；跟踪脚本显式传入 `0.1, 10`。

输出文件：

| 文件 | 内容 |
| --- | --- |
| `tracking.csv` | 每 1 ms 的参考/实际状态、力矩、更新标记和时钟滞后 |
| `tracking.png` | 六关节参考与实际位置曲线、位置误差曲线 |
| `torque_hold.png` | 六关节力矩阶梯图，显示零阶保持效果 |
| `mpc_updates.csv` | 每个后台任务的状态采样时刻、目标执行时刻、完成时刻、耗时及是否采用 |
| `summary.json` | 完成状态、QP 失败和错过更新次数、跟踪误差、耗时及实际运行时间 |

关节位置误差使用 deg；`tool_site` 位置误差使用 mm，是当前 XML 中该点的
欧氏距离误差（参考与实际均由该模型的正运动学求得），不包含末端姿态误差。

`mpc_updates.csv` 的 `solve_ms` 包含延迟状态预测、MPC 预测、线性化和 QP，
从任务开始运行到生成结果计时；`finished_wall_s` 是相对仿真启动墙钟的完成时间。
统计以每个任务为单位，超时阈值是 MPC 周期（默认 100 ms）。
`initial_solve_ms` 单独记录启动前求解时间。

`tracking.csv` 的 `solve_ms` 仅在收取任务结果的那一行非零；`torque_updated`
标记是否切换到新力矩（初始力矩也标记为 1）。
`cycle_ms` 是快循环本身的计算耗时，不包含等待、画面刷新和后台计算。
`wall_lag_ms` 记录快循环比计划时钟晚了多久。

对比 `simulated_time_s` 与 `wall_elapsed_s` 可以判断是否接近正常动画速度。
即使平均速度接近真实时间，也不代表每次调用都严格准时：应同时查看时钟滞后。
预测补偿使用同一套机器人参数，结果仍不能代表存在模型误差的实机性能。
旧目录 `mpc_tracking` 和 `mpc_tracking_1000hz` 对应之前的同步求解实验。

接口参考：[pybind11 CMake 文档](https://pybind11.readthedocs.io/en/stable/compiling.html)、
[MuJoCo Python viewer 文档](https://mujoco.readthedocs.io/en/stable/python.html)。
