# v3：先准备模型，再用最新测量反馈

## 当前状态

代码已经写入，但**尚未在 Ubuntu 编译、运行 C++ 数值测试或测量频率**。
没有连接板卡或机器人。以下命令由你执行，不代表已经测试通过。
原有 `src_v2`、`include_v2`、项目根目录 CMake 和 Python 脚本没有改动。

## 1. 最简单的理解

- `prepare()`：较慢。在未来模型块的预测状态附近计算动力学、A/B、缺陷和 QP，提前冷启动求解器。
- `feedback()`：较快。只用最新测量更新状态误差和 QP 梯度，然后热启动得到新力矩。不调用ABA，不等待PCIe。
- 后台准备没完成、模型偏差过大、测量过期或QP失败时，反馈报错，不把旧结果当新结果。

默认 `model_dt=0.01`、`feedback_ratio=3`、`horizon=25`：

| 反馈tick | 逻辑时刻 | 使用的模型 | 首预测区间 |
|---|---|---|---|
| 0 | 0 ms | 第0块，第0相位 | 10 ms |
| 1 | 3.333… ms | 第0块，第1相位 | 6.666… ms |
| 2 | 6.666… ms | 第0块，第2相位 | 3.333… ms |
| 3 | 10 ms | 第1块，第0相位 | 10 ms |

第0块的未来状态节点始终位于10、20、…、250 ms。块内不是每次移位10 ms；
每个相位单独准备首段模型，预测起点随反馈时刻前进，其他未来节点保持原来的绝对时刻。
因此本块内剩余预测时长为250、246.667、243.333 ms，下一块恢复250 ms。
这是有意选择的“固定未来节点、缩短首区间”方案，不是假装三个周期相同，也不是严格恒定长度的滚动网格。

**300 Hz只是目标：必须同时满足反馈<3.333ms和后台下一块及时准备好。**
后台还要计算三个首段模型和冷启动三个QP，这不是免费工作。
本版是“约100Hz模型准备+约300Hz测量反馈”，不是完整NMPC每秒重算300次。

## 2. 数学变化

参考v2继续使用Pinocchio解析导数和同一小步积分公式。
为了批量计算不同预测区间，名义状态不再必须来自一条串行的、严格可行的轨迹。
每个区间都计算动力学缺陷：

```text
d_i = F_i(x_bar_i, u_bar_i) - x_bar_(i+1)
delta_x_(i+1) = A_i*delta_x_i + B_i*delta_u_i + d_i
DeltaX = Phi*delta_x0 + Gamma*DeltaU + eta
```

`eta`递推为`eta_(i+1)=A_i*eta_i+d_i`，代价函数和输出的预测状态都包含它。
最新测量通过`delta_x0=measurement-x_bar0`进入每次反馈QP。
这与v2中`delta_x0=0、缺陷=0`的简化不同，不能遗漏。

每个相位的H保持固定，所以用固定H的`QProblemB::hotstart(g,lb,ub,...)`是正确的。
新模型块构造新QP对象，H的生命周期覆盖求解器，不在旧对象内原地改H。
每次反馈返回的是新测量对应的控制，不是发送三次同一个力矩。

Q/F/R与v2一致。为CPU/FPGA对照，默认力矩限幅改为±10Nm，另加名义力矩增量信赖范围±5Nm。
CPU可通过Config修改限幅；当前FPGA部署域不允许超过±10Nm。
这些默认值不是安全认证，比较v2时应使用相同限幅。

## 3. 文件对应关系

| 实现 | 头文件 | 作用 |
|---|---|---|
| `mpc_controller_v3.cpp` | `include_v3/mpc_controller_v3.h` | prepare/feedback、完整模型发布、时间与误差检查 |
| `prepared_model.cpp` | `include_v3/prepared_model.h` | 三个相位、缺陷、Phi/Gamma/eta、QP代价 |
| `prepared_qp.cpp` | `include_v3/prepared_qp.h` | 准备时冷启动，反馈时固定H热启动 |
| `dynamics_backend.cpp` | `include_v3/dynamics_backend.h` | CPU批量区间积分接口 |
| `python_bindings.cpp` | `include_v3/python_api_v3.h` | Python接口，释放GIL支持准备线程 |
| `test_v3.cpp` | — | 可手算的双积分器数学/调度测试 |
| `demo_pipeline.py` | — | CPU/mock/FPGA使用同一虚拟对象与时序的对照入口 |

动力学直接复用 `src_v2/pinocchio_fun.cpp` 与 `include_v2`，不复制修改旧实现。
FPGA版本链接本版控制核心，只替换准备阶段的 `DynamicsBackend`。

## 4. Ubuntu构建（手动执行）

把四个新增目录连同原有v2、URDF放在同一个项目根目录下，先激活你现有环境。
不要替换根目录CMake；单独构建：

```bash
cmake -S src_v3 -B build_v3 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v3 -j2
ctest --test-dir build_v3 --output-on-failure
```

依赖沿用v2：Eigen3、Pinocchio、qpOASES、pybind11、Python开发头文件。
CMake选错Python时指定`-DPython3_EXECUTABLE=/你的环境/bin/python3`。
不需要Python时加`-DBUILD_PYTHON_BINDINGS=OFF`。
模块输出为`build_v3/python/rokae_mpc_v3*.so`，不会覆盖原模块。

## 5. 先验证数值，再测调度

演示程序只使用Pinocchio虚拟对象、小幅正弦参考，不接机器人、不自动启动硬件。
先运行数值模式：

```bash
PYTHONPATH="$PWD/build_v3/python" python3 src_v3/demo_pipeline.py --backend cpu --mode numerical
```

`numerical`模式允许在模型块边界等待准备结果，且不强制反馈墙钟截止时间。
用于检查数学与轨迹，不可用于声称达到300Hz。

然后运行按墙钟调度的模式：

```bash
PYTHONPATH="$PWD/build_v3/python" python3 src_v3/demo_pipeline.py --backend cpu --mode paced
```

`paced`模式每3.333…ms读取虚拟状态并重新求解；准备或反馈错过截止时间就停止。
没有静默跳过、沿用旧模型或重复发送旧力矩的策略。
它测的是主机调度和求解能力：虚拟对象按离散网格演进，没有模拟实际机器人通信及执行延迟，
不是完整硬件在环测试，也不证明闭环稳定性。

结果写入`src_v3/results/cpu_numerical`或`cpu_paced`。
`feedback.csv`记录反馈墙钟时间、QP时间、状态与力矩；`preparation.csv`记录后台成本。
重复运行会覆盖同名结果文件，可以用`--output 新目录`保存不同实验。
第一块模型在计时前准备，但其耗时仍记录在preparation第0行。

建议比较`--ratio 1`与`--ratio 3`，而不是仅比较打印出的目标Hz。
关注：完整反馈耗时、准备是否按时、失败次数、相同参考下的跟踪误差。

## 6. 如何在你自己的程序里使用

```python
from concurrent.futures import ThreadPoolExecutor
from rokae_mpc_v3 import Config, MPCController

cfg = Config()  # 先修改Config，再创建控制器；创建后配置固定。
mpc = MPCController(urdf_path, cfg)
mpc.prepare(0, initial_state, refs_block0, ddq_block0)

# 真正的周期调度和测量由你的调用方负责：
result = mpc.feedback(latest_state, tick=0, measurement_age_s=actual_age)
tau = result.torque
# 结果仍须经过你的机器人安全层；这里没有机器人发送函数。

# 一个常驻线程准备下一块；result.states[1]恰好对应下一块起点。
worker = ThreadPoolExecutor(max_workers=1)
future = worker.submit(mpc.prepare, 1, result.states[1].copy(), refs_block1, ddq_block1)
# 主线程在tick 1、2继续用新测量调用feedback，不必等待future。
# tick 3前必须确认下一块已完成，并处理future中的异常。
```

完整、会回收线程和检查异常的示例在`demo_pipeline.py`。
`state_ref`形状为`(N+1,12)`，`ddq_ref`形状为`(N,6)`。
第b块参考的时间是`(b+i)*model_dt`，不是反馈tick乘以整个model_dt。
调用方应在新实验/时间轴归零之前调用`reset()`。

## 7. 限制与实物接入要求

- 缓存只在对应模型块内使用，不允许过期块自动延长寿命。
- 测量偏差和名义动力学缺陷使用可配置阈值；它们是工程门限，不是稳定性或鲁棒性证明。
- 测量年龄必须由真实时间戳计算。默认0只是为了仿真方便，不允许实机伪造新鲜测量。
- tick代表逻辑调度时间；C++不替你检测外部调度延迟。示例paced模式另检查绝对墙钟deadline。
- C++反馈预算包含函数主要工作和锁等待，不包含全部Python返回、设备通信和实际施加力矩时间；外层仍要计时。
- 模型准备使用预测初始状态；扰动过大时会拒绝，需要调用方触发重准备或经验证的安全回退。
- 只有力矩边界和局部力矩信赖范围，没有关节位置/速度/碰撞/力矩变化率约束。
- 同一控制器支持一个prepare线程与一个feedback线程；各自串行，不共享Pinocchio Data。
- 两个线程仍竞争CPU和内存带宽，不承诺三倍提速。必须记录准备阶段的真实成本。
- 默认假设底层补偿重力，仍须核对真实设备的接口、安装方向、负载和模型。
- 实机通信、急停、看门狗、限幅和超时回退必须在接入层实现。本版绝不直接发送机器人命令。

先读本说明，再读`src_fpga/README.md`。两套控制器的差别是后台动力学，不是两套不同的QP数学模型。
