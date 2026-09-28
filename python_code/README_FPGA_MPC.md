# FPGA 批量 ABA 的 MPC 仿真控制器

新增文件，不修改 `track_mpc.py`、`src/`、URDF、XML、Verilog 或 bit 文件。
运行平台为 Ubuntu，使用已加载 bit 的 XDMA 板卡，不使用 XRT/xclbin，不需要编译本项目的 C++ 扩展。
本程序只向 MuJoCo 施加力矩，不连接真实机器人；尚未完成板上闭环验证。

## 先看这个兼容性问题

当前原 MPC 的 `src/pinocchio_fun.cpp` 把重力设为零，MuJoCo 也使用零重力。
本地 FPGA 源码 `E:/20260824_FPGA_ABA/20260811_ABA/src_HLS/pass3.cpp` 的基座加速度却包含 `9.81`。
新控制器在每批中计算并相减：

```
ddq_zero(q,dq,u) = ABA_earth(q,dq,u) - ABA_earth(q,0,0)
```

对于同一个刚体模型，这能消去重力项，无需修改 bit。定点误差可能影响相减结果，所以仍需检查精度。
`--fpga-gravity zero` 只能用于你确认实际位流已经取消重力的情况，不能用于绕过错误。

**此外，本地 HLS 和 URDF 的机器人参数/坐标定义并不等价。**
按 HLS 的 `T_R_out.cpp`、`v_ori.cpp`、`ABA_parms.cpp` 中变换和空间惯量，以及
URDF 的关节变换、轴向、质心和惯量，分别以 `M = sum(J_i.T @ I_i @ J_i)` 作双精度静态计算：

| 同一关节坐标 | HLS 模型 M11 | 当前 URDF M11 |
|---|---:|---:|
| 六关节零位 | 1.825059 | 0.472022 |
| 当前轨迹起点 | 3.683010 | 1.444886 |

这是本地源代码的模型检查，不是板上测量，也不是逐位定点仿真。不同坐标系的参数不能只按文字逐项比较；
上表比较的是按各自变换组合后的关节空间惯性矩阵，差异远大于舍入误差。
如果板上 bit 来自这份源码，预期启动校验会失败；不能声称它已经可以正确替代当前 URDF 的 ABA。

下一步先运行下方 `--check-only`。如果不通过：

1. 确认板上 bit 对应的实际 HLS 版本及外部定点格式。
2. 对齐机器人关节顺序、正方向、零位偏置、MDH/URDF 坐标变换以及变换后的惯量/质心。
3. 必要时由你重新生成匹配模型的 IP/bit，再运行校验；本次没有改写或编译硬件工程。
4. 不要靠放宽误差阈值、改用浮点解释或关闭检查来强行通过。

## 文件职责

| 文件 | 作用 |
|---|---|
| `fpga_aba.py` | 常驻 XDMA 句柄、定点转换、mailbox 握手、CPU 对照后端 |
| `fpga_mpc.py` | 批量差分线性化、单次 SQP、固定稀疏结构的 OSQP |
| `track_mpc_fpga.py` | 校验模型、读轨迹、MuJoCo 仿真、性能/误差日志 |
| `test_fpga_mpc.py` | 不依赖板卡、Pinocchio、MuJoCo 的软件单元测试 |

## 为什么适合这块流水线

原代码不是简单调用 15 次 ABA：当 `period=0.1, horizon=15` 时，每段用 100 个 1 ms 小步积分，
再对 12 个状态和 6 个控制量做前向差分。因此名义轨迹加差分约需
`15 * (1+18) * 100 = 28500` 次 ABA；后台的提前状态预测还另需约 100 次。
原 `Prediction.cpp` 每周期重新构造 qpOASES 问题。

新控制周期为：

1. 将上一周期优化的状态/力矩轨迹移位，作为当前名义轨迹；首周期用参考轨迹初始化。
2. 每节点生成名义样本及 18 个变量的正负扰动，共 37 组。
3. 对含重力的 bit，每节点另加 13 个不同 q 的零速零力矩样本，消除重力。
4. 默认 N=15，共 `15*(37+13)=750` 组，一次 H2C、一次 start/done、一次 C2H。
5. 组装线性时变动力学约束，用 OSQP 解一个 QP，执行第一段力矩。

没有强行补齐 1000 组；无效样本不会提高有效吞吐。含重力的 bit 最多 N=20（1000 组），
零重力 bit 最多 N=27（999 组）。每周期参考力矩 RNEA、QP 和仿真仍由 CPU 完成。
重力消除只适用于模型里的重力项；不能补偿机器人模型本身不一致。

### 必须理解的算法差异

这不是把旧代码里的 ABA 函数逐个换成 PCIe 调用。原来沿预测时域依赖前一步结果的 rollout，
无法一次把未知未来状态全部送进流水线。这里改为 multiple-shooting 的一次 SQP 更新：

```
x[k+1] = A[k] x[k] + B[k] u[k] + c[k]
c[k]   = F(xbar[k],ubar[k]) - A[k] xbar[k] - B[k] ubar[k]
```

`c[k]` 不可省略：移位后的名义状态通常不严格满足非线性动力学。
每段 `F` 使用一次常加速度积分：`q+=dt*dq+0.5*dt^2*ddq; dq+=dt*ddq`，
没有旧版的 1 ms 子步循环；大步长、高速度或强非线性下精度可能下降。
没有收敛到完整非线性最优解的保证，需要仿真验证跟踪误差和约束。

默认 `period=0.01`，目标 100 Hz，而不是原来的 10 Hz；N=15 对应预测窗 0.15 s，
原来是 1.5 s。这是不同的预测窗，不能把全部性能变化归功于 FPGA。
若需要保持原来的 1.5 s 预测窗且控制 100 Hz，需要进一步设计非均匀网格或多速率控制，
不能简单把本版本 N 改成 150（会超过单批容量）。

原权重保留为位置 100、速度 1、力矩修正 0.01、终端状态权重加倍。
原力矩限制 ±30 Nm；本版本默认 ±10 Nm，因为 HLS 头文件的已声明测试域是
q∈[-π,π]、dq∈[-2,2]、tau∈[-10,10]。外部 Q8.24 可表示 ±128 不代表内部运算在整个范围都可靠。
可以用 `--torque-limit 30` 做对照，但必须先通过该范围的数值校验；运行中还会检查输出饱和。
QP 限制状态在 URDF 与保守 HLS q/dq 范围的交集，并给差分保留余量。
这些线性预测约束不等同于真实机器人安全保证。

## Ubuntu 安装：仅使用预编译包

把新增文件放到 Ubuntu 同一项目的 `python_code/`，保持上一级 `urdf/`、`xml/`、`stl/`、`data_in/` 目录。
以下假设项目在 `/home/yyy/Y2_1/20260831_ROBOT_NMPC`，实际路径不同时替换 `cd`。

```bash
cd /home/yyy/Y2_1/20260831_ROBOT_NMPC
python3 -m venv .venv-fpga
.venv-fpga/bin/python -m pip install --only-binary=:all: -r python_code/requirements_fpga.txt
```

使用 Python 3.10 或以上。`pin` 是机器人 Pinocchio 的 pip 包名，导入名是 `pinocchio`。
`--only-binary=:all:` 找不到匹配包会停止，不会偷偷执行本机源代码编译。
若缺少 venv 或缺少对应架构的 wheel，请先配置 Ubuntu 的 Python 环境，不要混用旧 `.so`。
本版本无需加载 `rokae_mpc.cpython-310-x86_64-linux-gnu.so`。

### 1. 不连接 FPGA，验证软件与 CPU 仿真

```bash
cd python_code
../.venv-fpga/bin/python -B -m unittest test_fpga_mpc -v
../.venv-fpga/bin/python -B track_mpc_fpga.py --backend cpu --duration 5
```

CPU 后端运行相同的新算法、同样数量的 ABA 样本，用于分离算法变化和 FPGA 收益。
这不是旧 C++ 控制器的性能测试，也不是 CPU 可达到的最佳解析导数实现。

### 2. 只检查 FPGA 动力学和差分精度

先确认 XDMA 驱动正常、bit 已加载，停止旧 `run_aba_xrt.py` 和其他板卡访问进程。

```bash
sudo ../.venv-fpga/bin/python -B track_mpc_fpga.py --backend fpga --check-only
```

依次检查原始带重力 ABA、消除重力后的 ABA、加速度对 q/dq/tau 的差分导数。
默认 q/dq 扰动 0.002，tau 扰动 0.01；旧 C++ 的 1e-6 差分步长不能照搬到定点流水线。
启动检验默认使用整条参考轨迹的 12 个位置及确定性的力矩探针，不代表覆盖所有可能预测状态。
`Raw ABA check` 的 normalized_max 必须 ≤1；Jacobian 的三个指标必须 ≤0.2。

若只有差分不通过，可以在检查模式分别比较 `--fd-q 0.005 --fd-dq 0.005 --fd-tau 0.02`；
误差随步长没有稳定区间时，应检查内部定点精度。若原始 ABA 不通过，先处理模型/格式问题。

### 3. 校验通过后运行 FPGA 仿真

```bash
sudo ../.venv-fpga/bin/python -B track_mpc_fpga.py \
  --backend fpga --period 0.01 --horizon 15 --duration 5
```

不带 `--viewer` 测性能；需要看运动时另加 `--viewer`。仿真不按墙钟时间限速，画面可能快于实时。
默认 `--poll-interval 0` 忙轮询以减少完成检测延迟，会占用 CPU；轮询仍包括 XDMA 驱动开销。
若 FPGA timeout/状态错误，程序停止且不自动重试；先人工检查/复位。驱动阻塞的系统调用不受 Python 轮询 timeout 硬性约束。

### 4. 如何判断频率确实提高

每次运行生成独立的 `data_out/fpga_mpc/<backend>_<时间戳>/`：

- `states.csv`：仿真时间、q、dq、实际输入力矩。
- `timing.csv`：参考力矩、批量线性化、QP 更新/求解、整个控制调用耗时；FPGA 还拆分传输和握手。

控制耗时统计排除第一次 QP setup，另行打印第一次耗时；观察样本数至少数百次。
`command_ms` 包含命令传输、板上执行、轮询检测，不是纯 FPGA 周期计数。
判断 100 Hz 的预算应看 **control_wall_ms 的 p99 和最大值是否低于 10 ms，并留出裕量**，
同时看 `Budget overruns` 和跟踪误差；不能用 `1000 / mean` 单独保证实时性。
终端的“求解时间倒数”只是计算能力指标，不是已经实现的真实闭环 Hz。

先以相同 N、period、力矩限制、时长分别运行 CPU/FPGA 后端，比较控制总耗时和跟踪误差。
若 QP 比批量计算更慢，缩短轮询间隔也不能解决该瓶颈。
只有精度和预算都通过后，才尝试 `--period 0.005`（目标 200 Hz，预测窗同时缩短到 0.075 s）。

重要：此入口为理想零计算延迟的同步仿真，求解期间 MuJoCo 不推进。
因此即使计算超时，模拟轨迹仍可能很好；这不能证明实机控制可行。
真实部署还需要加入传感/执行延迟、实时调度、独立底层伺服、安全停机、通信故障策略等；本程序不包含这些。

## 固定协议与故障边界

基地址 `0xC0000000`；command/status/count 相对偏移 `0x80/0x84/0x88`；
输入 `0x100`（每组 72 字节），输出 `0x12000`（每组 24 字节），最多 1000 组。
q/dq=Q4.28、tau=Q8.24、ddq=Q16.16，小端有符号 int32。
每批结束读出结果，再清 command 并确认 idle，防止下一批把上一批 done 当成新结果。
进程加协作式文件锁，但旧脚本未加锁，所以仍必须确保板卡独占。
不要改变输入/输出位置以尝试提速；这些位置由现有 RTL 决定。

## 本次验证范围

已在开发机执行 13 项纯软件单元测试，验证定点打包/解包、饱和拒绝、分段传输、mailbox 顺序、
忙状态/超时/错误拒绝、重力样本对应关系、离散雅可比、容量、QP 单步解析解、移位更新和失败处理。
测试使用解析玩具动力学与模拟 mailbox，不打开 XDMA。
未编译任何 C++、HLS 或 Vivado 工程，未访问板卡，未在 Ubuntu 上跑 Pinocchio/MuJoCo 闭环。
因此 100 Hz 是待测试的目标，不是实测结果；当前源码模型差异尤其需要先解决。

实现中使用的 OSQP 接口依据官方文档：
[Python update / warm start](https://osqp.org/docs/interfaces/python.html)。
固定稀疏结构允许更新数值而不重复 setup；A 的数值改变仍可能重新分解矩阵，不应理解为完全没有更新成本。
