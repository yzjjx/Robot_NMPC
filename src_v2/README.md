# CPU MPC v2：先把 Pinocchio 版本做快，再评估 FPGA

这一版是独立目录。原来的 `src`、`include`、项目 CMake 和 Python 脚本没有被替换。
Python 模块名为 `rokae_mpc_v2`，不会覆盖 `rokae_mpc`。

**当前交付状态：代码已写入，尚未在 Ubuntu 编译、执行数值测试或测量速度。没有连接机器人。**
不要将“文件已生成”理解为“已经通过实物验证”。以下命令由你在 Ubuntu 手动执行。

## 1. 先看懂做了什么

旧版每个预测区间：

1. 积分一条名义轨迹。
2. 再积分 12 条状态扰动轨迹和 6 条力矩扰动轨迹。
3. 用结果相减得到 A、B。

v2 每个预测区间：

1. 只沿名义轨迹积分。
2. 每个小步用 Pinocchio 解析导数计算加速度对 q、dq、tau 的导数。
3. 用链式法则递推整个区间的 A、B。

QP 对象持续保存，并使用支持 Hessian 变化的 `SQProblem::hotstart`。
固定权重只初始化一次；Gamma、QP 输入缓冲区重复使用。
这不是完全无内存分配的硬实时实现，Eigen/Pinocchio/qpOASES 内部仍可能分配内存。

以下内容刻意保持不变，方便对比：

- 每个预测区间内保持力矩不变；内部积分小步默认不超过 1 ms。
- 积分公式为 `q_next=q+h*dq+0.5*h*h*ddq`、`dq_next=dq+h*ddq`。
- Q：位置权重 100、速度权重 1；F=2Q；R=0.01I。
- 默认附加力矩约束 ±30 Nm，与旧程序一致，**不是已经验证的实物安全限值**。
- 从本次测量开始重新生成名义轨迹；只移位上一周期控制，不缓存状态/A/B。
- 每周期求解一个局部线性化 QP，不承诺每周期收敛到非线性最优解。

本版没有 FPGA、P0/P1 队列、模型有效性判定、跨周期状态缓存、碰撞约束或机器人 SDK。
先验证 CPU 计算和仿真，再接实物接口。不要直接把仿真循环用于机器人。

## 2. 文件阅读顺序

| 文件 | 作用 |
|---|---|
| `NMPC_control.cpp/.h` | 主流程：参考力矩、积分和导数、矩阵、QP、输出第一组力矩 |
| `pinocchio_fun.cpp/.h` | ABA、RNEA、状态积分、解析敏感度递推 |
| `MPC_Matrices.cpp/.h` | 构造 Gamma；对角权重不生成巨大的 Q_bar |
| `Prediction.cpp/.h` | 构造 QP、热启动、失败检查 |
| `types.h` | 6 关节的固定尺寸类型、计时字段 |
| `python_bindings.cpp` | 独立 Python 模块，锁保护 Pinocchio 可变数据 |
| `main.cpp` | 不连接机器人的 C++ 状态快照耗时测试 |
| `test_v2.cpp` | 导数、凝聚矩阵、变化 H 的热启动等数值检查 |
| `compare_cpu.py` | 相同快照、相同参数的新旧控制器对比 |
| `track_mpc_v2.py` | 简单同步 MuJoCo 仿真，无后台过期结果问题 |
| `CMakeLists.txt` | 独立 Ubuntu 构建配置 |

所有头文件都在本目录，不依赖旧 `include`。C++ API 使用 `mpc_v2` 命名空间，
不是旧类的二进制替换；不要只拷贝几个 cpp 到旧 CMake 中混编。

## 3. 数学上怎么从 ABA 导数得到 A、B？

设 g(q,v,u)=ddq。Pinocchio 给出 gq、gv、gu，其中 gu=M(q)^{-1}。
每个小步 h 的导数为：

```text
a = [ I + 0.5*h²*gq    h*I + 0.5*h²*gv ]
    [ h*gq             I + h*gv        ]

b = [ 0.5*h²*gu ]
    [ h*gu      ]
```

初始化 `A=I, B=0`，每个小步更新 `A=a*A, B=a*B+b`。
这才是完整区间积分映射的导数，不是把加速度导数直接当 A、B。
Minv 按上三角恢复成对称矩阵，以兼容仅填上三角的 Pinocchio 接口。

本版名义状态每次从最新测量积分，所以初始状态增量为零、动力学缺陷为零，
只需 `DeltaX=Gamma*DeltaU`。以后若加入状态缓存，必须补 Phi 和缺陷项，不能直接沿用此简化。

为接口清楚，本版每小步显式调用 ABA，再调用带 q/v/u 的完整解析导数接口。
解析导数仍有成本；不是把计算量无代价地缩小 19 倍。

## 4. 在 Ubuntu 构建（由你执行）

将整个 `src_v2` 放在 Ubuntu 项目根目录下，与 `src`、`urdf`、`xml`、`data_in` 同级。
先激活你原来能够运行 Pinocchio、MuJoCo 的环境。以下命令都在项目根目录执行。
依赖为 C++17、CMake、Eigen3、Pinocchio、qpOASES；Python 接口还需要 pybind11 和 Python 开发头文件。

```bash
cmake -S src_v2 -B build_v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2 -j2
ctest --test-dir build_v2 --output-on-failure
```

依赖不在标准位置时，向 CMake 传入已有安装的 `CMAKE_PREFIX_PATH` 或相应包的 `_DIR`，
不要覆盖旧构建目录。若 CMake 选错 Python，可指定 `-DPython3_EXECUTABLE=/你的环境/bin/python3`。
本目录的构建默认 Release，不使用 `-ffast-math`。

只编译 C++、不需要 Python 时：

```bash
cmake -S src_v2 -B build_v2_cpp -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=OFF
cmake --build build_v2_cpp -j2
ctest --test-dir build_v2_cpp --output-on-failure
```

测试包括：

- 0、1、10、100 ms 区间的解析 A/B 与中心差分比较，开/关重力两种模型。
- 带导数积分与普通积分产生同一下一状态。
- Gamma 与逐节点扰动传播一致。
- Hessian 连续变化、上下界切换时，热启动与冷启动解一致。
- 控制器失败标志、非法输入、失败后重新初始化。

这些测试需要你编译后执行；不等同于稳定性证明或机器人安全测试。

## 5. 先测计算速度，不接机器人

```bash
./build_v2/mpc_benchmark urdf/ROKAE_SR4.urdf 0.1 15 100
```

它用固定状态快照，输出平均/P99/最大计算时间及阶段时间，排除前 5 次预热。
实际运行时首周期也要计入预算；快照测试不能代替轨迹和实物测量。

Python 对比新旧版本（旧 `rokae_mpc` 必须已可导入，且也使用 Release 构建）：

```bash
PYTHONPATH="$PWD/build_v2/python:$PWD/python_code" python3 src_v2/compare_cpu.py --old --dt 0.1 --horizon 15 --count 100
```

只有 v2 模块时去掉 `--old`。两个版本使用相同输入快照序列，输出力矩差异供检查；
旧版是前向差分，v2 是解析导数，不要求浮点逐位相等。
这个脚本不自动判定实物可用，不会执行机器人动作。

## 6. 再做 MuJoCo 跟踪

先使用和原脚本相同的 0.1 秒、15 步，不同时改变算法与预测范围：

```bash
PYTHONPATH="$PWD/build_v2/python" python3 src_v2/track_mpc_v2.py --period 0.1 --horizon 15 --duration 3
```

加 `--viewer` 显示窗口。默认不显示窗口，减小渲染对计时的影响。
结果写到 `src_v2/results/tracking_v2.csv` 和 `timing_v2.csv`。
重复运行同一个输出目录会覆盖这两个结果文件，可用 `--output 新目录` 分开实验。

本脚本从当前仿真测量同步求解，不再预先计算下一时刻力矩；求解失败直接停止。
**求解期间仿真时间暂停，因此它只验证数值跟踪与计算耗时，不模拟通信延迟，
也不证明你达到了实时控制频率。** 原脚本的异步延迟处理不同，不能仅凭两条仿真曲线归因。
每周期 wall_ms 包含 Python 调用；C++ 的 total_ms 不包含机器人通信。

确认数值正确后才评估更短周期，例如 0.01 秒。15 步此时预测范围从 1.5 秒变成 0.15 秒，
不能把这部分计算量下降全部算成代码优化收益。
本版仍保持“控制更新周期=预测节点间隔”，没有偷偷改变移位含义。

## 7. 你自己的 Python 程序如何调用？

```python
from rokae_mpc_v2 import MPCController

mpc = MPCController(urdf_path, timestep=0.01, horizon=15,
                    gravity_compensated=True, integration_step=0.001)
# state: (12,), state_ref: (16,12), ddq_ref: (15,6)
tau = mpc.compute_control(state, state_ref, ddq_ref)
print(mpc.timing.dynamics_ms, mpc.timing.qp_ms, mpc.timing.total_ms)
```

这里 0.01 秒只是接口示例，不是已经通过测试的实物频率。
`mpc.set_torque_limits(lower, upper)` 可设置六轴各自上下界，设置后重新冷启动。
`warm_start=False` 可创建解析导数+冷启动版本，用于分离两项优化收益。
`mpc.reset()` 用于新实验开始或控制中断后，清除旧控制序列。

C++ 接入对应的类是 `mpc_v2::Dynamics` 和 `mpc_v2::Controller`。
Dynamics 必须比 Controller 活得更久，且同一个实例不得并发调用。
Python 包装层有互斥保护，但不会让同一控制器自动变成硬实时。

## 8. 实物前不可省略的事情

- 核实接口发送的是附加力矩还是总力矩。默认 `gravity_compensated=True` 与旧程序一致，
  仅在底层确实补偿重力时成立；False 使用 URDF 构建模型的默认重力，还需核实安装方向。
- 检查模型、关节顺序/符号、速度、惯量、工具负载、单位与真实设备一致。
- 按设备规定设置逐轴限幅。当前仅有力矩上下界，没有位置/速度/碰撞/力矩变化率约束。
- 使用带时间戳的状态和控制输出，建立最大结果年龄、看门狗和截止时间处理。
- 本版异常不会返回“假成功”的零力矩，但真正的停止/回退策略必须由机器人接口实现。
- 均值/P99/观测最大值都不等于最坏执行时间证明，QP 的迭代上限也不等于墙钟截止时间。
- 不要在机器人通信回调里打印、写 CSV、渲染窗口或无限等待 QP。

## 9. 以后接 FPGA

先保存优化后的 CPU 基线，再评估将独立 ABA 任务批量提交给 FPGA。
本版的解析导数留在 CPU，仍需要 Pinocchio 内部数据；不能简单把 FPGA 的 ddq
写入 Data 后就认为解析导数准备完成。解析导数可能已经让 CPU 足够快，FPGA 净收益需要重测。

参考接口：

- Pinocchio：<https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/coverage/index.include_pinocchio_algorithm_aba-derivatives.hpp.html>
- qpOASES SQProblem：<https://www.coin-or.org/qpOASES/doc/3.1/doxygen/classSQProblem.html>
