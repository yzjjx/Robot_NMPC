# src 与 src_v2 逐文件差异说明

本文比较当前工作区中的 `src/` 和 `src_v2/`，方向为 **src → src_v2**，不是比较 Git 历史版本。对照日期：2026-09-29。

每个文件先说明接口、实现与行为变化，再给出机器生成的完整 unified diff。`-` 行属于 `src`，`+` 行属于 `src_v2`，空格开头的行是上下文；`@@` 标注两侧行号。所有发生变化的代码、注释和空行都包含在 diff 中，未变化的大段内容只保留邻近上下文。比较时统一 CRLF/LF 换行符，不把换行格式转换算作代码修改。

本文只读取代码，没有编译、运行控制器或测量性能。说明中引用 `include/` 和 `include_v2/` 的定义，是为了核实类型、默认参数与求解器声明；头文件本身不属于本次逐行 diff 范围。`src_v3/`、`src_fpga/`、Python 仿真、实机入口和 CMake 也不在本次比较范围。

diff 中的 `\ No newline at end of file` 表示前一行在原文件中没有结尾换行符，它是差异格式的标记，不是 C++ 代码。本次 `src/MPC_Matrices.cpp` 存在这种情况，已保留标记。

## 1. 文件清单与整体变化

| 文件 | 状态 | src 行数 | src_v2 行数 | 新增行（+） | 移除行（-） |
|---|---|---:|---:|---:|---:|
| `MPC_Matrices.cpp` | 修改 | 48 | 36 | 31 | 43 |
| `NMPC_control.cpp` | 修改 | 230 | 99 | 92 | 223 |
| `Prediction.cpp` | 修改 | 104 | 86 | 78 | 96 |
| `main.cpp` | 仅 src 存在 | 132 | 0 | 0 | 132 |
| `pinocchio_fun.cpp` | 修改 | 151 | 105 | 85 | 131 |
| `python_bindings.cpp` | 修改 | 90 | 90 | 70 | 70 |

文件总数：src 有 6 个，src_v2 有 5 个。同名文件共 5 对，全部有修改；仅 src 存在 1 个文件，仅 src_v2 存在 0 个文件。

行数包含注释和空行。新增、移除行数按本文的 unified diff 统计，不代表修改的函数数量。

### 1.1 阅读顺序

1. [pinocchio_fun.cpp：动力学与导数](#file-dynamics)
2. [MPC_Matrices.cpp：预测矩阵](#file-matrices)
3. [Prediction.cpp：QP 求解](#file-prediction)
4. [NMPC_control.cpp：控制流程](#file-controller)
5. [python_bindings.cpp：Python 接口](#file-python)
6. [main.cpp：仅 src 中存在的示例入口](#file-main)

### 1.2 两版共同完成什么任务

输入当前状态 `x = [q; dq]`，其中 `q` 是 6 个关节角度，`dq` 是 6 个关节速度；再输入未来的参考状态和参考加速度。控制器预测未来运动，求解一个有力矩上下限的二次规划问题，输出本周期使用的 6 个关节力矩。

两版都保留以下设计：

- 6 个关节，状态维度 12，控制维度 6。
- 用 RNEA 根据参考状态和加速度生成参考力矩。
- 第一次用参考力矩初始化名义控制序列，之后将上次最优控制序列前移一位，末项重复。
- 名义轨迹从当前实际状态开始，所以初始偏差 `delta_x0 = 0`。
- 每个预测区间保持力矩不变，用多个小积分步推进状态。
- 使用线性化后的模型构造 QP，只执行最优力矩序列的第一项。
- 默认不包含重力项，力矩表示机器人底层重力补偿之外的附加力矩。
- 默认位置权重 100、速度权重 1、终端权重为普通状态权重的 2 倍、控制权重 0.01。
- 默认力矩范围仍为每个关节 `[-30, 30] N·m`，不是 v2 自动改成了 10 N·m。Python 示例可以另行设置限幅。
- QP Hessian 的对角正则项仍为 `1e-6`，求解迭代上限仍为 500。

主要变化是求导、内存复用、矩阵构造、求解器状态保存和接口组织；没有在 v2 中新增 PD、FPGA 或 v3 的跨周期模型准备机制。

### 1.3 调用流程对照

```text
src:
compute_control
  → RNEA 参考力矩
  → 初始化 / 前移名义力矩
  → generate_nom_traj：积分名义轨迹
  → calculate_Mat_AB：对状态和力矩逐维扰动，再重复积分
  → compute_mpc_matrices：Phi、Gamma、Q_bar、R_bar
  → Prediction：新建 QProblemB，冷启动求解
  → 保存优化后的控制序列和线性预测状态序列
  → 返回第一组力矩

src_v2:
compute_control
  → RNEA 参考力矩，并初始化 / 前移名义力矩
  → integrate_linearized：同一条名义轨迹上积分并传播解析导数
  → MPCMatrices.update：Gamma、按行加权的 Gamma
  → Prediction.prepare：构造本周期 QP
  → Prediction.solve：复用 SQProblem，允许 Hessian 变化的热启动
  → 保存优化后的控制序列
  → 返回第一组力矩，并记录分阶段耗时
```

<a id="file-dynamics"></a>

## 2. pinocchio_fun.cpp

源文件：[src/pinocchio_fun.cpp](src/pinocchio_fun.cpp) → [src_v2/pinocchio_fun.cpp](src_v2/pinocchio_fun.cpp)。

### 2.1 接口对应关系

| src | src_v2 | 含义 |
|---|---|---|
| `pinocchioFun` | `mpc_v2::Dynamics` | 动力学类改名并放入命名空间 |
| 构造函数只接收 URDF 路径 | 还接收 `gravity_compensated`、`integration_step` | 可配置重力约定与积分小步长 |
| `set_forward_diff_step` | 无对应方法 | 解析求导不再需要设置差分步长 |
| `compute_rnea(q, dq, ddq)` | `inverse_dynamics(x, ddq)` | q、dq 改为通过一个状态向量传入 |
| `compute_aba(x, u, Ts)` | `acceleration(x, u)` 与 `advance(x, ddq, h)` | 将加速度计算与状态推进分开 |
| `compute_held_state(x, u, duration)` | `integrate(x, u, duration)` | 固定力矩下积分一段时间 |
| `com_Mat_A_B(...)` | `integrate_linearized(x, u, duration)` | 输出一个区间的终态及 A、B，外层控制器循环处理各区间 |
| 无独立函数 | `step_count(duration)` | 集中计算并检查积分小步数 |

旧 `compute_aba` 的名字容易让人以为它只返回加速度，但它实际返回的是 **12 维下一时刻状态**。v2 的 `acceleration` 才是直接返回 **6 维加速度** 的函数。

### 2.2 逐项实现与行为变化

| 项目 | src | src_v2 |
|---|---|---|
| 导数依赖 | 使用 ABA、RNEA | 新增 `aba-derivatives.hpp`，调用 `computeABADerivatives` |
| 模型维度 | 设置 `DOF = 6`，构造时未验证 URDF 的 nq/nv | 构造时检查 nq=nv=6，逐个检查标量关节 |
| 重力 | 无条件 `model.gravity.setZero()` | 仅 `gravity_compensated=true` 时置零；默认 true，默认行为相同 |
| 小步长 | 固定不超过 0.001 s | 使用可配置的 `max_step_`，要求有限且大于 0 |
| 积分步数 | `max(1, ceil(duration/0.001))` | `max(1, ceil(duration/max_step_))`，额外限制不超过 1,000,000 步 |
| duration=0 | 仍进入一次 `compute_aba`，计算加速度后乘零 | 普通积分直接返回原状态；线性化返回原状态、单位 A、零 B |
| 向量类型 | `Eigen::VectorXd`，运行时检查长度 | `State`、`Joint` 固定为 12、6 维，定义见 `include_v2/types.h` |
| 输入校验 | 检查维度、有限值和时间参数；差分步长 setter 未校验 | 固定维度配合有限值检查，集中检查时间、小步数、模型结构 |
| 输出校验 | ABA/RNEA 返回后未在该函数内显式检查有限性 | 检查加速度、逆动力学力矩、积分结果、A 和 B 的有限性 |
| 临时数据 | 分离 q/dq、创建扰动向量，多次重复积分 | 直接用状态分块，使用固定尺寸局部矩阵 |
| A、B 生成 | 调用者预先准备数组，函数逐列填充 | 返回 `Linearization{next, A, B}`，控制器保存各区间结果 |

局部类名、成员名及 include 调整，以及注释删改，全部列在本节末尾的 diff 中。

### 2.3 为什么不再需要逐维扰动

旧版把输入中的一个量增加很小的数，再重新模拟整段运动，用两次输出的差估计变化率。状态 12 维、力矩 6 维，所以每个预测区间需要 **12+6=18 次扰动积分**，此外还需要名义轨迹的积分。

v2 在每个小步直接取得加速度的导数：

```text
gq = ∂ddq/∂q
gv = ∂ddq/∂dq
gu = ∂ddq/∂u = M(q)^(-1)
```

两版状态推进公式相同：

```text
q_next  = q + h*dq + 0.5*h²*ddq
dq_next = dq + h*ddq
```

v2 对这个公式求导，得到当前小步的 a、b：

```text
a = [ I + 0.5*h²*gq    h*I + 0.5*h²*gv ]
    [       h*gq              I + h*gv ]

b = [ 0.5*h²*gu ]
    [      h*gu ]
```

随后用链式法则把所有小步串起来：`A_new = a*A_old`，`B_new = a*B_old + b`。因此不是只对最后一个小步求导，也不是简单使用 `I+h*Ac`。`gu` 从 `Minv` 的上三角恢复成对称矩阵，避免依赖下三角是否已被库填充。

### 2.4 ABA 数量如何变化

设预测区间数为 N，每个区间包含 K 个积分小步，在输入有效且正常完成一次求解的情况下：

| 操作 | src | src_v2 |
|---|---:|---:|
| 参考力矩 RNEA | N 次 | N 次 |
| 名义推进和导数阶段显式 `pinocchio::aba` 调用 | `19*N*K` 次 | `N*K` 次 |
| `computeABADerivatives` 调用 | 0 次 | `N*K` 次 |
| 12+6 组扰动轨迹 | 每区间计算 | 不再计算 |

例如 N=25、预测区间 10 ms、小步 1 ms 时，旧版显式 ABA 为 4750 次；v2 为 250 次显式 ABA，加 250 次完整导数接口调用。这里没有把外部 `predict_state` 的额外积分计入 MPC 求解。

**不能因此直接说 v2 快了 19 倍。** 完整 `computeABADerivatives(model,data,q,v,u)` 本身也有动力学计算成本，v2 没有采用只复用上次 ABA 中间量的接口；QP、矩阵运算、内存与 Python 调用也会影响总耗时。

### 2.5 完整逐行 diff

```diff
--- src/pinocchio_fun.cpp
+++ src_v2/pinocchio_fun.cpp
@@ -1,151 +1,105 @@
 #include "pinocchio_fun.h"
 
 #include <algorithm>
-#include <cmath>
-#include <stdexcept>
+#include <pinocchio/parsers/urdf.hpp>
+#include <pinocchio/algorithm/aba.hpp>
+#include <pinocchio/algorithm/aba-derivatives.hpp>
+#include <pinocchio/algorithm/rnea.hpp>
 
-#include <pinocchio/algorithm/rnea.hpp>
-#include <pinocchio/algorithm/aba.hpp>
-#include <pinocchio/parsers/urdf.hpp>
+namespace mpc_v2 {
 
-// 需要在h文件新建一个pinocchioFun的类class
-// 这个类有三个核心功能，分别为rnea计算、aba计算与矩阵计算
-
-// 首先为构造函数，构造函数是创建一个对象时，自动执行的一段初始化程序
-pinocchioFun::pinocchioFun(
-    const std::string& urdf_path)
-    :model(),data(model)
-{
-    DOF = 6; // 自由度
-    q_step = 1e-6; // 前向差分计算步长
-    dq_step = 1e-6;
-    tau_step = 1e-4;
-    
-    pinocchio::urdf::buildModel(urdf_path, model);
-
-    // 机器人底层始终补偿重力；RNEA、ABA及其线性化统一使用附加力矩。
-    // tau = M(q) * ddq + C(q, dq) * dq，不再包含 g(q)。
-    model.gravity.setZero();
-    data = pinocchio::Data(model);
+Dynamics::Dynamics(const std::string& urdf, bool gravity_compensated, double integration_step)
+    : data_(model_), max_step_(integration_step) {
+    require(std::isfinite(max_step_) && max_step_ > 0, "integration_step must be positive");
+    pinocchio::urdf::buildModel(urdf, model_);
+    require(model_.nq == DOF && model_.nv == DOF, "Expected a 6-DOF fixed-base model");
+    // q+v*h 只适用于这里的六个标量关节，不支持浮动基座/四元数关节。
+    for (std::size_t i = 1; i < model_.joints.size(); ++i)
+        require(model_.joints[i].nq() == 1 && model_.joints[i].nv() == 1,
+                "Expected scalar joints");
+    // 与旧版默认相同：输入为机器人底层补偿重力之后的附加力矩。
+    // 只有确认接口接收总力矩时，才把 gravity_compensated 设为 false。
+    if (gravity_compensated) model_.gravity.setZero();
+    data_ = pinocchio::Data(model_);
 }
 
-// 构造函数的部分扩展，构造函数为初始化参数，这部分的函数可以修改构造函数的大小
-void pinocchioFun::set_forward_diff_step(
-    double q_step_,
-    double dq_step_,
-    double tau_step_)
-{
-    q_step = q_step_;
-    dq_step = dq_step_;
-    tau_step = tau_step_;
+int Dynamics::step_count(double duration) const {
+    require(std::isfinite(duration) && duration >= 0, "duration must be finite and nonnegative");
+    const double count = std::ceil(duration / max_step_);
+    require(count <= 1000000, "Too many integration substeps");
+    return std::max(1, static_cast<int>(count));
 }
 
-// RNEA计算
-Eigen::VectorXd pinocchioFun::compute_rnea(
-    const Eigen::VectorXd& q,
-    const Eigen::VectorXd& dq,
-    const Eigen::VectorXd& ddq)
-{
-    // 检查输入维度是否正确
-    if (q.size() != DOF || dq.size() != DOF || ddq.size() != DOF) {
-        throw std::invalid_argument("输入维度不正确.");
-    }
-    if (!q.allFinite() || !dq.allFinite() || !ddq.allFinite()) {
-        throw std::invalid_argument("RNEA inputs must be finite.");
-    }
-
-    // 使用Pinocchio计算RNEA得到关节力矩
-    Eigen::VectorXd tau = pinocchio::rnea(model, data, q, dq, ddq);
-    
-    return tau;
+State Dynamics::advance(const State& x, const Joint& ddq, double h) {
+    State next;
+    next.head<DOF>() = x.head<DOF>() + h*x.tail<DOF>() + (0.5*h*h)*ddq;
+    next.tail<DOF>() = x.tail<DOF>() + h*ddq;
+    return next;
 }
 
-
-// ABA计算，ABA是根据当前q、dq和tau输出ddq
-Eigen::VectorXd pinocchioFun::compute_aba(
-    const Eigen::VectorXd& state,
-    const Eigen::VectorXd& control,
-    double Ts)
-{
-    // 检查输入维度是否正确
-    if (state.size() != 2 * DOF || control.size() != DOF) {
-        throw std::invalid_argument("输入维度不正确.");
-    }
-    if (!state.allFinite() || !control.allFinite() || !std::isfinite(Ts) || Ts < 0) {
-        throw std::invalid_argument("ABA inputs must be finite and timestep nonnegative.");
-    }
-
-    // 将状态向量拆分为关节位置和速度
-    Eigen::VectorXd q = state.head(DOF);
-    Eigen::VectorXd dq = state.tail(DOF);
-
-    // 使用Pinocchio计算ABA得到关节加速度，control就是tau
-    Eigen::VectorXd ddq = pinocchio::aba(model, data, q, dq, control);
-
-    // 按当前加速度作常加速度离散化，得到下一时刻的状态
-    Eigen::VectorXd next_state(2 * DOF);
-    next_state.head(DOF) = q + dq * Ts + 0.5 * ddq * Ts * Ts; // 更新关节位置
-    next_state.tail(DOF) = dq + ddq * Ts; // 更新关节速度
-
-    return next_state;
+Joint Dynamics::acceleration(const State& x, const Joint& u) {
+    require(x.allFinite() && u.allFinite(), "Nonfinite ABA input");
+    Joint ddq = pinocchio::aba(model_, data_, x.head<DOF>(), x.tail<DOF>(), u);
+    if (!ddq.allFinite()) throw std::runtime_error("Nonfinite ABA result");
+    return ddq;
 }
 
-
-// 一个预测区间内保持力矩不变，以不超过 1 ms 的步长积分
-// 给定当前状态和一个固定力矩，预测duration秒后机器人到哪里
-// 在一个预测区间内，控制输入保持不变，即零阶保持器
-Eigen::VectorXd pinocchioFun::compute_held_state(
-    const Eigen::VectorXd& state,
-    const Eigen::VectorXd& control,
-    double duration)
-{
-    if(!std::isfinite(duration) || duration < 0) {
-        throw std::invalid_argument("Prediction duration not ok");
-    }
-    // ceil是向上取整
-    const int steps = std::max(1, static_cast<int>(std::ceil(duration / 0.001)));
-    // 预测加速度ddq
-    Eigen::VectorXd predicted = state;
-    for(int i = 0; i < steps; ++i) {
-        predicted = compute_aba(predicted, control, duration / steps);
-    }
-    return predicted;
+Joint Dynamics::inverse_dynamics(const State& x, const Joint& ddq) {
+    require(x.allFinite() && ddq.allFinite(), "Nonfinite RNEA input");
+    Joint u = pinocchio::rnea(model_, data_, x.head<DOF>(), x.tail<DOF>(), ddq);
+    if (!u.allFinite()) throw std::runtime_error("Nonfinite RNEA result");
+    return u;
 }
 
-// 矩阵计算
-void pinocchioFun::com_Mat_A_B(
-    const std::vector<Eigen::VectorXd>& nom_state,
-    const std::vector<Eigen::VectorXd>& nom_control,
-    const std::vector<Eigen::VectorXd>& nom_state_next,
-    double Ts,
-    std::vector<Eigen::MatrixXd>& Mat_A,
-    std::vector<Eigen::MatrixXd>& Mat_B
-)
-{
-    int N = static_cast<int>(nom_control.size()); // 预测步长
-    int x_n = nom_state[0].size(); // 状态维度12
-    int u_n = nom_control[0].size(); // 控制输入维度6
+State Dynamics::integrate(const State& x, const Joint& u, double duration) {
+    require(x.allFinite() && u.allFinite(), "Nonfinite integration input");
+    const int steps = step_count(duration);
+    const double h = duration / steps;
+    State current = x;
+    if (duration == 0) return current;
+    for (int j = 0; j < steps; ++j)
+        current = advance(current, acceleration(current, u), h);
+    if (!current.allFinite()) throw std::runtime_error("Nonfinite integrated state");
+    return current;
+}
 
-    // 取出当前这个预测步长的状态矩阵（12）和控制输入（6）
-    for (int i = 0; i < N; i++) {
-        const Eigen::VectorXd& x_bar = nom_state[i];
-        const Eigen::VectorXd& u_bar = nom_control[i];
-        const Eigen::VectorXd& x_bar_next = nom_state_next[i];
+Linearization Dynamics::integrate_linearized(const State& x, const Joint& u, double duration) {
+    require(x.allFinite() && u.allFinite(), "Nonfinite linearization input");
+    const int steps = step_count(duration);
+    const double h = duration / steps;
+    Linearization out{x, MatA::Identity(), MatB::Zero()};
+    if (duration == 0) return out;
+    const Eigen::Matrix<double, DOF, DOF> I = Eigen::Matrix<double, DOF, DOF>::Identity();
 
-        for(int j = 0; j < x_n; j++) {
-            const double step = (j < DOF) ? q_step : dq_step;
-            Eigen::VectorXd x_perturbed = x_bar;
-            x_perturbed(j) += step; // 位置、速度分别使用各自的差分步长
-            Eigen::VectorXd x_next_perturbed = compute_held_state(x_perturbed, u_bar, Ts);
-            Mat_A[i].col(j) = (x_next_perturbed - x_bar_next) / step; // 计算雅可比矩阵A的每一列
-        }
+    for (int j = 0; j < steps; ++j) {
+        // 先保存加速度，再计算同一个 (q,v,u) 点的解析导数。
+        // 使用带 q/v/u 的完整接口，避免依赖不同 Pinocchio 版本的中间量复用约定。
+        const Joint ddq = acceleration(out.next, u);
+        pinocchio::computeABADerivatives(model_, data_, out.next.head<DOF>(),
+                                       out.next.tail<DOF>(), u);
+        const Eigen::Matrix<double, DOF, DOF> gq = data_.ddq_dq;
+        const Eigen::Matrix<double, DOF, DOF> gv = data_.ddq_dv;
+        // 某些版本只填 Minv 的上三角，必须恢复对称矩阵。
+        const Eigen::Matrix<double, DOF, DOF> gu = data_.Minv.selfadjointView<Eigen::Upper>();
 
-        for(int j = 0; j < u_n; j++) {
-            Eigen::VectorXd u_perturbed = u_bar;
-            u_perturbed(j) += tau_step; // 对控制输入向量的每个元素进行扰动
-            Eigen::VectorXd x_next_perturbed = compute_held_state(x_bar, u_perturbed, Ts);
-            Mat_B[i].col(j) = (x_next_perturbed - x_bar_next) / tau_step; // 计算雅可比矩阵B的每一列
-        }
-        
+        // 对当前常加速度积分公式求导，而不是简单使用 I+h*Ac。
+        MatA a;
+        a.topLeftCorner<DOF, DOF>() = I + (0.5*h*h)*gq;
+        a.topRightCorner<DOF, DOF>() = h*I + (0.5*h*h)*gv;
+        a.bottomLeftCorner<DOF, DOF>() = h*gq;
+        a.bottomRightCorner<DOF, DOF>() = I + h*gv;
+        MatB b;
+        b.topRows<DOF>() = (0.5*h*h)*gu;
+        b.bottomRows<DOF>() = h*gu;
+
+        // 链式法则：从小步的导数得到整个保持力矩区间的导数。
+        out.A = (a*out.A).eval();
+        out.B = (a*out.B + b).eval();
+        out.next = advance(out.next, ddq, h);
+        if (!out.next.allFinite() || !out.A.allFinite() || !out.B.allFinite())
+            throw std::runtime_error("Nonfinite state or sensitivity");
     }
+    return out;
 }
+
+} // namespace mpc_v2
```

<a id="file-matrices"></a>

## 3. MPC_Matrices.cpp

源文件：[src/MPC_Matrices.cpp](src/MPC_Matrices.cpp) → [src_v2/MPC_Matrices.cpp](src_v2/MPC_Matrices.cpp)。

### 3.1 接口及存储变化

| 项目 | src | src_v2 |
|---|---|---|
| 入口 | 自由函数 `compute_mpc_matrices(A,B,Q,R,F,N)` | `mpc_v2::MPCMatrices` 类，构造时给 N，随后 `update(A,B)` |
| 返回方式 | 每次返回新建的 `MPC_Matrices` 结构体 | 更新对象内可复用的成员矩阵 |
| Phi | 每次构造并保存 | 移除，当前控制器始终有 `delta_x0=0` |
| Gamma | 每次分配并计算 | 构造时分配，每次覆盖更新 |
| Q_bar、R_bar | 建立完整大矩阵，包含很多零 | 使用 `state_weights`、`control_weights` 保存对角元素 |
| 加权 Gamma | QP 阶段计算 `Gammaᵀ*Q_bar*Gamma` | 提前按行计算 `weighted_Gamma = Q_bar*Gamma` |
| 临时行块 | `Gamma_i = A[i]*Gamma_i` | `row_`、`next_row_` 两块缓冲，`noalias()` 计算后交换 |
| Phi 递推 | 额外维护 `Phi_i` | 移除对应计算 |
| 权重设置 | 函数支持传入一般的 Q、R、F 矩阵 | 构造函数中固定为当前默认的对角权重 |
| 形状约束 | 从第一个矩阵取尺寸，函数内无 A/B 数组长度检查 | 固定 12/6 维，检查 A、B 数组长度等于 N |

因此，v2 的目标权重与 **当前 src 控制器的默认配置** 一致，但新矩阵类不再提供旧自由函数那种任意 Q/R/F 输入能力。若将来要使用非对角权重，需要修改 v2 的数据结构与构造方式。

### 3.2 保留的数学关系

两版仍在构造下三角块矩阵 Gamma：第 i 个未来状态只受当前及更早的控制增量影响。v2 删除 Phi 的前提是名义起点等于当前测量状态，这一前提在 `Controller::compute_control` 中由 `x_[0] = state` 保证。

若未来改为复用旧名义起点，使 `delta_x0 != 0`，就不能直接沿用“完全不计算 Phi”的处理方式。

优化主要减少无用零元素的存储、权重矩阵的重复构造和 Phi 运算。Gamma 与 QP Hessian 仍然是大矩阵，不能说 v2 已变成稀疏 QP 或不再有动态内存分配。

### 3.3 完整逐行 diff

```diff
--- src/MPC_Matrices.cpp
+++ src_v2/MPC_Matrices.cpp
@@ -1,48 +1,36 @@
 #include "MPC_Matrices.h"
 
-MPC_Matrices compute_mpc_matrices(
-    const std::vector<Eigen::MatrixXd>& Mat_A,
-    const std::vector<Eigen::MatrixXd>& Mat_B,
-    const Eigen::MatrixXd& Q,
-    const Eigen::MatrixXd& R,
-    const Eigen::MatrixXd& F,
-    int N
-)
-{
-    int n = Mat_A[0].rows(); // 状态维度
-    int p = Mat_B[0].cols(); // 控制输入维度
+namespace mpc_v2 {
 
-    // Delta X = phi*Delta_x0+Gamma*Delta_U
-    // 这里n = 12， p = 6
-    Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(N*n, n);
-    Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(N*n, N*p);
+MPCMatrices::MPCMatrices(int horizon) : N_(checked_horizon(horizon)) {
+    Gamma.setZero(N_*NX, N_*DOF);
+    weighted_Gamma.resize(N_*NX, N_*DOF);
+    row_.setZero(NX, N_*DOF);
+    next_row_.resize(NX, N_*DOF);
+    state_weights.resize(N_*NX);
+    control_weights.setConstant(N_*DOF, 0.01);
+    // 与旧版一致：Q=diag(100,...,100,1,...,1)，终端 F=2Q。
+    for (int i = 0; i < N_; ++i) {
+        const double terminal = (i == N_-1) ? 2.0 : 1.0;
+        state_weights.segment(i*NX, DOF).setConstant(100.0*terminal);
+        state_weights.segment(i*NX+DOF, DOF).setConstant(terminal);
+    }
+}
 
-    Eigen::MatrixXd Phi_i = Eigen::MatrixXd::Identity(n, n);
-    Eigen::MatrixXd Gamma_i = Eigen::MatrixXd::Zero(n, N*p);
+void MPCMatrices::update(const std::vector<MatA>& A, const std::vector<MatB>& B) {
+    require(static_cast<int>(A.size()) == N_ && static_cast<int>(B.size()) == N_,
+            "A/B horizon mismatch");
+    row_.setZero();
+    for (int i = 0; i < N_; ++i) {
+        next_row_.noalias() = A[i]*row_;
+        next_row_.block(0, i*DOF, NX, DOF) += B[i];
+        row_.swap(next_row_);
+        Gamma.middleRows(i*NX, NX) = row_;
+    }
+    // 对角权重直接乘每一行，不构造巨大的 Q_bar。
+    weighted_Gamma = Gamma;
+    for (int r = 0; r < Gamma.rows(); ++r)
+        weighted_Gamma.row(r) *= state_weights(r);
+}
 
-    for(int i = 0; i < N; i++){
-        //delta_x_i = A_i*delta_x_{i}+B_i*delta_u_i
-        Phi_i = Mat_A[i] * Phi_i;
-        Gamma_i = Mat_A[i] * Gamma_i;
-        Gamma_i.block(0, i*p, n, p) += Mat_B[i];
-        
-        Phi.block(i*n, 0, n, n) = Phi_i;
-        Gamma.block(i*n, 0, n, N*p) = Gamma_i;
-    }
-
-    // 状态权重使用Q，终端使用F，也就是说Q_bar的最后一行使用F
-    Eigen::MatrixXd Q_bar = Eigen::MatrixXd::Zero(N*n, N*n);
-    for(int i = 0; i < N-1; i++){
-        Q_bar.block(i*n, i*n, n, n) = Q;
-    }
-    Q_bar.block((N-1)*n, (N-1)*n, n, n) = F;
-
-    // 控制权重：R
-    Eigen::MatrixXd R_bar = Eigen::MatrixXd::Zero(N*p, N*p);
-    for(int i = 0; i < N; i++){
-        R_bar.block(i*p, i*p, p, p) = R;
-    }
-
-    return {Phi, Gamma, Q_bar, R_bar};
-
-}
\ No newline at end of file
+} // namespace mpc_v2
```

<a id="file-prediction"></a>

## 4. Prediction.cpp

源文件：[src/Prediction.cpp](src/Prediction.cpp) → [src_v2/Prediction.cpp](src_v2/Prediction.cpp)。

### 4.1 QP 数学目标基本保持

设 `e = nominal_X - ref_X`，`du` 是整段控制增量，v2 与旧版在 `delta_x0=0` 时使用相同形式：

```text
H = 2*(Gammaᵀ*Q_bar*Gamma + R_bar) + 1e-6*I
g = 2*(Gammaᵀ*Q_bar*e + R_bar*(nominal_U-ref_U))
lower = tau_lower - nominal_U
upper = tau_upper - nominal_U
```

旧版在函数内部还允许通过 `Phi*delta_x0` 修正 e；v2 由控制器直接提供 e，不再接收 Phi、起点偏差及单独的参考状态向量。

### 4.2 逐项差异

| 项目 | src | src_v2 |
|---|---|---|
| 组织形式 | 一个 `Prediction(...)` 自由函数 | 持久化 `mpc_v2::Prediction` 对象，分成构造、`prepare`、`solve`、`reset` |
| 求解器 | 每次新建 `QProblemB` | 对象持有 `SQProblem`，零个一般线性约束，仅保留变量边界 |
| 初始化 | 每次 `init` | 首次或关闭热启动时 `init`，已初始化且启用热启动时 `hotstart` |
| Hessian 变化 | 每次新问题直接使用新 H | 使用可接收新 H 的 `SQProblem::hotstart` 重载 |
| 配置 | 无求解器热启动开关 | `warm_start` 参数控制求解器热启动 |
| 工作内存 | 每次新建 H、g、边界及解向量 | 构造时预分配工作向量和两份 H |
| 行优先转换 | `Eigen2QpArray` 双层循环，逐元素复制到 `std::vector` | 直接使用 `qpOASES::real_t` 类型的行优先 Eigen 矩阵，传 `.data()` |
| H 缓冲 | 当次函数局部存储 | `H_[0]`、`H_[1]` 轮流使用，避免准备新 H 时覆盖求解器仍引用的旧 H |
| H 构造 | 显式加 R_bar 和单位矩阵 | 先计算加权 Gamma 乘积，再逐个对角元素添加控制权重和正则项 |
| g 构造 | 完整权重矩阵乘法 | 权重向量逐元素乘误差，再乘 Gamma 转置 |
| 力矩边界 | 每个预测区间复制 6 维上下界 | 按索引 `i % DOF` 选择对应关节上下界 |
| 输入检查 | 检查构造后的 H/g/边界有限性 | 还检查误差和控制向量长度，使用 `prepared_` 检查调用顺序 |
| 迭代上限 | 局部 `nWSR=500` | 成员 `iterations_=500`，求解后可读取实际使用值 |
| 求解失败 | 打印错误码，返回向量，用 `qp_success` 标志通知上层 | 清空求解器有效状态并抛异常，不返回伪装成成功的结果 |
| 解检查 | 成功标志要求解向量有限 | 额外按 `1e-5` 容差检查上下界是否满足 |
| 返回 | 按值返回 `Eigen::VectorXd` | 返回成员 `delta_` 的 const 引用，避免一次返回拷贝 |
| 再次求解 | 每次重新创建局部对象 | 失败后必须重新 prepare；下一次求解走冷启动 |

`SQProblem` 的具体成员类型、行优先类型和 H 双缓冲声明在 [include_v2/Prediction.h](include_v2/Prediction.h) 中。v2 显式禁止该 Prediction 对象的拷贝，避免复制含内部指针和状态的求解器；这项声明来自头文件，不是 `.cpp` 中新增的函数。

### 4.3 容易误解的地方

- 旧版顶层 `compute_control` 也会检查失败标志并抛异常，因此不能说旧版一定会把失败 QP 的零增量发送给机器人。
- “前移上次最优力矩”与“QP 热启动”是两件事。两版都有前者，v2 新增的是后者；把 `warm_start=false` 不会关闭控制序列前移。
- H 双缓冲用于维护 qpOASES 引用的数据生命周期，不表示同时求解两个 QP。
- v2 的 `prepare/solve` 是同一次 `compute_control` 内的两个阶段，不等于 v3 的后台准备、前台反馈架构。
- 返回引用在对象下一次修改工作缓冲后会改变，调用者若长期保存结果需要自行复制。当前控制器立即读取并写入自己的最优控制序列。

### 4.4 完整逐行 diff

```diff
--- src/Prediction.cpp
+++ src_v2/Prediction.cpp
@@ -1,104 +1,86 @@
 #include "Prediction.h"
+#include <string>
 
-#include <qpOASES.hpp>
-#include <iostream>
-#include <stdexcept>
-#include <vector>
+namespace mpc_v2 {
 
-// Eigen默认为列优先，而qpOASES要求输入为行优先，因此需要将Eigen矩阵转换为行优先存储
-template<typename Derived>//表明函数为模板函数，可以接受多种Eigen类型的输入
-std::vector<qpOASES::real_t> Eigen2QpArray(
-    const Eigen::MatrixBase<Derived>& eigen_matrix
-)
-{
-    std::vector<qpOASES::real_t> data;
-    data.reserve(eigen_matrix.size());
-
-    for (int i = 0; i < eigen_matrix.rows(); i++) {
-        for (int j = 0; j < eigen_matrix.cols(); j++) {
-            data.push_back(static_cast<qpOASES::real_t>(eigen_matrix(i, j)));
-        }
-    }
-    return data;
+Prediction::Prediction(int horizon, bool warm_start)
+    : n_(checked_horizon(horizon)*DOF), solver_(n_, 0), warm_start_(warm_start) {
+    for (auto& h : H_) h.resize(n_, n_);
+    g_.resize(n_);
+    lower_.resize(n_);
+    upper_.resize(n_);
+    solution_.resize(n_);
+    weighted_error_.resize(horizon*NX);
+    delta_.resize(n_);
+    qpOASES::Options options;
+    options.printLevel = qpOASES::PL_NONE;
+    solver_.setOptions(options);
 }
 
-Eigen::VectorXd Prediction(
-    const Eigen::VectorXd& delta_x0,
-    const Eigen::VectorXd& nominal_X,
-    const Eigen::VectorXd& nominal_U,
-    const Eigen::VectorXd& ref_X,
-    const Eigen::VectorXd& ref_U,
-    const MPC_Matrices& mpc_matrices,
-    const Eigen::VectorXd& tau_lower,
-    const Eigen::VectorXd& tau_upper,
-    int N,
-    int p,
-    bool* qp_success
-) 
-{
-    if (qp_success) {
-        *qp_success = false;
+void Prediction::reset() {
+    solver_.reset();
+    initialized_ = false;
+    prepared_ = false;
+    active_ = -1;
+}
+
+void Prediction::prepare(const MPCMatrices& m, const Eigen::VectorXd& error,
+                         const Eigen::VectorXd& nominal_u, const Eigen::VectorXd& reference_u,
+                         const Joint& lower, const Joint& upper) {
+    require(error.size() == m.Gamma.rows() && nominal_u.size() == n_ && reference_u.size() == n_,
+            "QP dimension mismatch");
+    prepared_ = false;
+    pending_ = (active_ == 0) ? 1 : 0;
+    auto& h = H_[pending_];
+    // 与旧版相同的目标函数、正则化；不降低精度、不放松力矩上下界。
+    h = (2.0*m.Gamma.transpose()*m.weighted_Gamma).cast<qpOASES::real_t>();
+    for (int i = 0; i < n_; ++i)
+        h(i, i) += static_cast<qpOASES::real_t>(2.0*m.control_weights(i) + 1e-6);
+    weighted_error_ = m.state_weights.cwiseProduct(error);
+    delta_.noalias() = m.Gamma.transpose()*weighted_error_;
+    delta_ += m.control_weights.cwiseProduct(nominal_u-reference_u);
+    g_ = (2.0*delta_).cast<qpOASES::real_t>();
+    for (int i = 0; i < n_; ++i) {
+        lower_(i) = static_cast<qpOASES::real_t>(lower(i % DOF)-nominal_u(i));
+        upper_(i) = static_cast<qpOASES::real_t>(upper(i % DOF)-nominal_u(i));
     }
-    int nV = N*p;
+    if (!h.allFinite() || !g_.allFinite() || !lower_.allFinite() || !upper_.allFinite())
+        throw std::runtime_error("Nonfinite QP data");
+    prepared_ = true;
+}
 
-    // 如果名义起点不等于实际测量状态，phi*delta_x0会修正所有状态预测
-    Eigen::VectorXd state_error = nominal_X + mpc_matrices.Phi * delta_x0 - ref_X;
-    Eigen::VectorXd control_err = nominal_U - ref_U;
+const Eigen::VectorXd& Prediction::solve() {
+    require(prepared_, "Call prepare before solve");
+    prepared_ = false;
+    iterations_ = 500; // 上限，不是每次必须迭代 500 次；不宣称硬实时。
+    qpOASES::returnValue status;
+    if (initialized_ && warm_start_) {
+        status = solver_.hotstart(H_[pending_].data(), g_.data(),
+                                 static_cast<const qpOASES::real_t*>(nullptr),
+                                 lower_.data(), upper_.data(), nullptr, nullptr, iterations_);
+    } else {
+        solver_.reset();
+        status = solver_.init(H_[pending_].data(), g_.data(),
+                              static_cast<const qpOASES::real_t*>(nullptr),
+                              lower_.data(), upper_.data(), nullptr, nullptr, iterations_);
+    }
+    if (status == qpOASES::SUCCESSFUL_RETURN)
+        status = solver_.getPrimalSolution(solution_.data());
+    if (status != qpOASES::SUCCESSFUL_RETURN || !solution_.allFinite()) {
+        reset();
+        throw std::runtime_error("QP failed, qpOASES code=" + std::to_string(static_cast<int>(status)));
+    }
+    delta_ = solution_.cast<double>();
+    // 失败时不把零向量/上一周期结果伪装成有效解。
+    const double tolerance = 1e-5;
+    if ((solution_-lower_).minCoeff() < -tolerance ||
+        (upper_-solution_).minCoeff() < -tolerance) {
+        reset();
+        throw std::runtime_error("QP torque bounds violated");
+    }
+    initialized_ = true;
+    active_ = pending_;
+    return delta_;
+}
 
-    // qpOASES格式
-    Eigen::MatrixXd H_qp = 2.0 * (mpc_matrices.Gamma.transpose()*mpc_matrices.Q_bar*mpc_matrices.Gamma+mpc_matrices.R_bar);
-    H_qp += 1e-6*Eigen::MatrixXd::Identity(nV,nV);
-
-    Eigen::MatrixXd g_qp = 2.0 * (mpc_matrices.Gamma.transpose()*mpc_matrices.Q_bar*state_error+mpc_matrices.R_bar*control_err);
-
-    // 实际力矩计算
-    Eigen::VectorXd lb = Eigen::VectorXd::Zero(nV);
-    Eigen::VectorXd ub = Eigen::VectorXd::Zero(nV);
-    for(int i = 0;i<N;i++)
-    {
-        lb.segment(i*p,p) = tau_lower-nominal_U.segment(i*p,p);
-        ub.segment(i*p,p) = tau_upper-nominal_U.segment(i*p,p);
-    }
-    if (!H_qp.allFinite() || !g_qp.allFinite() || !lb.allFinite() || !ub.allFinite()) {
-        throw std::runtime_error("QP matrices and bounds must be finite.");
-    }
-
-    std::vector<qpOASES::real_t> H_arr = Eigen2QpArray(H_qp);
-    std::vector<qpOASES::real_t> g_arr = Eigen2QpArray(g_qp);
-    std::vector<qpOASES::real_t> lb_arr = Eigen2QpArray(lb);
-    std::vector<qpOASES::real_t> ub_arr = Eigen2QpArray(ub);
-
-    qpOASES::QProblemB problem(nV);
-    qpOASES::Options options;
-    options.printLevel = qpOASES::PL_NONE;
-    problem.setOptions(options);
-
-    int nWSR = 500;
-    qpOASES::returnValue status = problem.init(
-        H_arr.data(),
-        g_arr.data(),
-        lb_arr.data(),
-        ub_arr.data(),
-        nWSR
-    );
-
-    Eigen::VectorXd delta_U = Eigen::VectorXd::Zero(nV);
-
-    if(status == qpOASES::SUCCESSFUL_RETURN)
-    {
-        std::vector<qpOASES::real_t> solution(nV);
-        status = problem.getPrimalSolution(solution.data());
-        if (status == qpOASES::SUCCESSFUL_RETURN) {
-            using QpVector = Eigen::Matrix<qpOASES::real_t, Eigen::Dynamic, 1>;
-            delta_U = Eigen::Map<const QpVector>(solution.data(), nV).cast<double>();
-            if (qp_success) {
-                *qp_success = delta_U.allFinite();
-            }
-        }
-    }
-    if(status != qpOASES::SUCCESSFUL_RETURN) {
-        std::cerr<<"求解失败，错误码："<<static_cast<int>(status)<<std::endl;
-    }
-
-    return delta_U;
-}
+} // namespace mpc_v2
```

<a id="file-controller"></a>

## 5. NMPC_control.cpp

源文件：[src/NMPC_control.cpp](src/NMPC_control.cpp) → [src_v2/NMPC_control.cpp](src_v2/NMPC_control.cpp)。

### 5.1 外部接口变化

| 项目 | src | src_v2 |
|---|---|---|
| 类 | `ROKAE_NMPC` | `mpc_v2::Controller` |
| 动力学引用 | `pinocchioFun&` | `Dynamics&` |
| 构造参数 | 动力学、控制周期、预测步数 | 增加 `warm_start` 参数 |
| 当前状态类型 | 动态 `VectorXd` | 固定 12 维 `State` |
| 参考状态 | `vector<VectorXd>`，N+1 个 12 维向量 | 矩阵，N+1 行、12 列 |
| 参考加速度 | `vector<VectorXd>`，N 个 6 维向量 | 矩阵，N 行、6 列 |
| 输出力矩 | 动态 `VectorXd` | 固定 6 维 `Joint` |
| 重置方法 | 无公开 reset | 新增 `reset()` |
| 限幅设置 | 构造函数固定 ±30，未提供公开 setter | 新增 `set_torque_limits(lower,upper)`；默认值仍为 ±30 |
| 状态查询 | `qp_success()` | `timing()` 中包含 `qp_success` 及耗时信息；getter 定义在头文件 |

### 5.2 内部流程逐项差异

| 项目 | src | src_v2 |
|---|---|---|
| 参数检查 | 时间有限且正、N≥1 | 时间检查保留，`checked_horizon` 在分配前要求 1≤N≤500 |
| 维度管理 | 构造函数保存 DOF、x_n、u_n | 统一使用 `include_v2/types.h` 的常量和固定类型 |
| 权重初始化 | 在控制器构造函数中创建 Q/F/R | 移到 MPCMatrices 构造函数中以对角权重形式保存 |
| 参考力矩容器 | 每次新建 `control_ref` | 使用预分配 `reference_u_` |
| 初始化与移位 | 分别调用 `initial_nom_ctrl` 和 `shift_nom_ctrl` | 合并到 RNEA 参考力矩循环中，用 `first_` 和 `min(i+1,N-1)` 选择 |
| 名义轨迹 | 单独调用 `generate_nom_traj` | 一次循环调用 `integrate_linearized`，同时获取 next/A/B |
| A/B | 单独调用 `calculate_Mat_AB`，执行差分 | 直接保存解析线性化返回值 |
| 名义状态缓存 | 同时保留 `nominal_state` 和 `nominal_state_next` | 使用 `x_`，不再重复保存同一组下一时刻状态 |
| 大向量 | 每次拼接 nominal_X、nominal_U、ref_X、ref_U | 使用可复用控制向量及 `state_error_`，直接计算每步状态误差 |
| 初始偏差 | 显式求 `current_state-nominal_state[0]`，结果为零 | 不再建立该零向量，也不计算对应 Phi 项 |
| QP 数据 | 局部创建 matrices，再调用自由函数 | 成员 `matrices_` 和 `qp_` 持续存在，先 update/prepare 再 solve |
| 输出控制序列 | 保存 `delta_tau` 和 `prediction_tau` | 直接生成并保存 `optimal_u_`，不单独保存每步 delta_tau |
| 最优状态序列 | 额外计算并保存线性修正后的 `prediction_state` | 移除该计算；仍有名义状态，但不保存旧版的优化后状态序列 |
| 最终力矩检查 | 检查 QP 标志和增量有限性 | 对每一步最终力矩检查有限性和上下界，容差 `1e-5` |
| 异常恢复 | 求解失败抛出；未统一重置首次周期状态 | try/catch 覆盖本次计算，任何异常都设置 `first_=true` 并重置 QP |
| 耗时记录 | 无分阶段记录 | 新增 steady_clock 计时和 Timing 数据 |

v2 的 `reset()` 清空首次周期标记、QP 状态和统计，不重新分配整套缓存。`set_torque_limits` 要求每个关节的上下界有限且下界严格小于上界，设置后调用 reset。控制器仍引用外部动力学对象，因此直接用 C++ 时，Dynamics 必须比 Controller 活得更久；Python 包装两版都通过成员持有解决此问题。

### 5.3 Timing 的含义

| 字段 | 当前代码记录的范围 |
|---|---|
| `reference_ms` | 开始计算到参考力矩与名义控制序列准备完成，包含输入检查 |
| `dynamics_ms` | 名义积分及解析线性化 |
| `matrices_ms` | Gamma 更新、QP 目标及上下界准备 |
| `qp_ms` | QP 求解阶段 |
| `total_ms` | 整次 C++ compute_control 时间，还包含结果处理等开销 |
| `qp_iterations` | 本次 QP 使用的迭代计数 |
| `qp_success` | 最终是否成功产生经过检查的控制序列 |

`total_ms` 不是 Python 整轮仿真时间，也不包括调用者在外部执行的 `predict_state`。异常时部分字段可能仍为零，不能把它当作对应阶段一定没有成本；失败路径至少会保存总耗时并将成功标志设为 false。

### 5.4 完整逐行 diff

```diff
--- src/NMPC_control.cpp
+++ src_v2/NMPC_control.cpp
@@ -1,230 +1,99 @@
 #include "NMPC_control.h"
-#include "pinocchio_fun.h"
-#include "MPC_Matrices.h"
-#include "Prediction.h"
+#include <algorithm>
+#include <chrono>
 
-#include <iostream>
-#include <cmath>
-#include <stdexcept>
-
-// 建立构造函数，给定control_dt为预测区间，例如ROKAE_NMPC nmpc(dynamics, 0.01, 20);即控制频率为10ms，100Hz，向未来预测20个区间，预测时间为200ms
-ROKAE_NMPC::ROKAE_NMPC(pinocchioFun& dynamics, double control_dt, int prediction_steps)
-    : dynamics(dynamics)
-{
-    // 控制器时间参数判断，finite：有限的
-    if(!std::isfinite(control_dt) || control_dt <= 0 || prediction_steps < 1) {
-        throw std::invalid_argument("MPC 时间参数不合法");
-    }
-
-    // 控制器输入基本参数，例如ROKAE_NMPC nmpc(dynamics,0.02,20),代表Ts为20ms，N为20
-    Ts = control_dt;  // 每段力矩保持的时间；可与底层发送周期不同
-    DOF = 6;    // 自由度
-    x_n = 12;   // 状态维度,分别为关节位置和关节速度
-    u_n = 6;    // 控制输入维度,关节力矩,这里控制输入指的是控制器输入到机器人里面的
-    N = prediction_steps;
-
-    first_control_cycle = true;// 标记是否为第一次控制周期
-
-    // 创建列矩阵
-    Eigen::VectorXd q_diag(x_n); // 状态权重矩阵对角线元素
-    q_diag.head(DOF) << 100, 100, 100, 100, 100, 100; // 关节位置权重
-    q_diag.tail(DOF) << 1, 1, 1, 1, 1, 1; // 关节速度权重
-    Q = q_diag.asDiagonal(); // 状态权重矩阵，将列矩阵转换为对角矩阵
-
-    F = 2.0*Q; // 终端状态权重矩阵，希望末端尽量接近目标
-    R = 0.01*Eigen::MatrixXd::Identity(u_n, u_n); // 控制输入权重矩阵，不希望力矩修正太大 
-
-    tau_lower = Eigen::VectorXd::Constant(u_n, -30); // 控制输入下界
-    tau_upper = Eigen::VectorXd::Constant(u_n, 30); // 控制输入上界
-
-    // 预测状态序列,21个12维状态向量，即nominal_state[0]一直到nominal_state[20]，每一个nominal_state都是一个12维的向量
-    nominal_state.resize(N+1, Eigen::VectorXd::Zero(x_n)); 
-    // 创建20个控制向量，20个6维控制向量
-    nominal_tau.resize(N, Eigen::VectorXd::Zero(u_n));
-
-    // 保存20次名义ABA计算得到的下一时刻状态的结果
-    nominal_state_next.resize(N, Eigen::VectorXd::Zero(x_n));
-
-    Mat_A.resize(N, Eigen::MatrixXd::Zero(x_n, x_n)); // 线性化状态矩阵
-    Mat_B.resize(N, Eigen::MatrixXd::Zero(x_n, u_n)); // 线性化控制矩阵
-
-    // 保存QP算出来的名义力矩改变量
-    delta_tau.resize(N, Eigen::VectorXd::Zero(u_n)); // 控制增量
-
-    prediction_tau.resize(N, Eigen::VectorXd::Zero(u_n)); // 预测控制输入
-
-    prediction_state.resize(N+1, Eigen::VectorXd::Zero(x_n)); // 预测状态序列，保存优化后预测的状态
+namespace mpc_v2 {
+namespace {
+using Clock = std::chrono::steady_clock;
+double milliseconds(Clock::time_point start, Clock::time_point end) {
+    return std::chrono::duration<double, std::milli>(end-start).count();
+}
 }
 
-// 第一个控制周期使用目标状态的RNEA力矩初始化U_bar,也就是预测区域内部20个参考状态通过RNEA计算得到的参考力矩
-void ROKAE_NMPC::initial_nom_ctrl(
-    const std::vector<Eigen::VectorXd>& control_ref
-)
-{
-    for(int i = 0; i < N; ++i) {
-        nominal_tau[i] = control_ref[i];
+Controller::Controller(Dynamics& dynamics, double timestep, int horizon, bool warm_start)
+    : dynamics_(dynamics), N_(checked_horizon(horizon)), Ts_(timestep),
+      x_(N_+1), u_(N_), optimal_u_(N_), A_(N_), B_(N_),
+      nominal_u_(N_*DOF), reference_u_(N_*DOF), state_error_(N_*NX),
+      matrices_(N_), qp_(N_, warm_start) {
+    require(std::isfinite(Ts_) && Ts_ > 0, "timestep must be positive");
+}
+
+void Controller::reset() {
+    first_ = true;
+    qp_.reset();
+    timing_ = Timing{};
+}
+
+void Controller::set_torque_limits(const Joint& lower, const Joint& upper) {
+    require(lower.allFinite() && upper.allFinite() && (lower.array() < upper.array()).all(),
+            "Expected finite lower < upper torque limits");
+    lower_ = lower;
+    upper_ = upper;
+    reset();
+}
+
+Joint Controller::compute_control(const State& state, const Eigen::MatrixXd& state_ref,
+                                 const Eigen::MatrixXd& ddq_ref) {
+    const auto start = Clock::now();
+    timing_ = Timing{};
+    try {
+        require(state_ref.rows() == N_+1 && state_ref.cols() == NX &&
+                ddq_ref.rows() == N_ && ddq_ref.cols() == DOF, "Reference shape mismatch");
+        require(state.allFinite() && state_ref.allFinite() && ddq_ref.allFinite(),
+                "Nonfinite controller input");
+
+        // 1. 与旧版一样，用参考轨迹的 RNEA 生成参考力矩。
+        for (int i = 0; i < N_; ++i) {
+            const State ref = state_ref.row(i).transpose();
+            const Joint ddq = ddq_ref.row(i).transpose();
+            reference_u_.segment<DOF>(i*DOF) = dynamics_.inverse_dynamics(ref, ddq);
+            u_[i] = first_ ? Joint(reference_u_.segment<DOF>(i*DOF))
+                          : optimal_u_[std::min(i+1, N_-1)];
+            nominal_u_.segment<DOF>(i*DOF) = u_[i];
+        }
+        const auto after_reference = Clock::now();
+        timing_.reference_ms = milliseconds(start, after_reference);
+
+        // 2. 只积分一条名义轨迹，同时得到每个区间的 A、B。
+        x_[0] = state;
+        for (int i = 0; i < N_; ++i) {
+            const auto step = dynamics_.integrate_linearized(x_[i], u_[i], Ts_);
+            x_[i+1] = step.next;
+            A_[i] = step.A;
+            B_[i] = step.B;
+            state_error_.segment<NX>(i*NX) = x_[i+1]-state_ref.row(i+1).transpose();
+        }
+        const auto after_dynamics = Clock::now();
+        timing_.dynamics_ms = milliseconds(after_reference, after_dynamics);
+
+        // 3. delta_x0=0，所以 delta_X = Gamma * delta_U。
+        matrices_.update(A_, B_);
+        qp_.prepare(matrices_, state_error_, nominal_u_, reference_u_, lower_, upper_);
+        const auto after_matrices = Clock::now();
+        timing_.matrices_ms = milliseconds(after_dynamics, after_matrices);
+
+        // 4. 求解器持续保存，H 变化时也能正确热启动。
+        const Eigen::VectorXd& delta_u = qp_.solve();
+        timing_.qp_ms = milliseconds(after_matrices, Clock::now());
+        timing_.qp_iterations = qp_.iterations();
+        for (int i = 0; i < N_; ++i) {
+            optimal_u_[i] = u_[i] + delta_u.segment<DOF>(i*DOF);
+            if (!optimal_u_[i].allFinite() || (optimal_u_[i]-lower_).minCoeff() < -1e-5 ||
+                (upper_-optimal_u_[i]).minCoeff() < -1e-5)
+                throw std::runtime_error("Invalid final torque sequence");
+        }
+        first_ = false;
+        timing_.qp_success = true;
+        timing_.total_ms = milliseconds(start, Clock::now());
+        return optimal_u_[0];
+    } catch (...) {
+        // 下次从参考重新初始化；当前必须把失败交给调用方处理。
+        first_ = true;
+        qp_.reset();
+        timing_.qp_success = false;
+        timing_.total_ms = milliseconds(start, Clock::now());
+        throw;
     }
 }
 
-// 第二个及其以后的控制周期：将上一个周期内的最优控制序列向前移动一位
-void ROKAE_NMPC::shift_nom_ctrl() {
-    for(int i = 0; i < N-1; ++i) {
-        nominal_tau[i] = prediction_tau[i+1];
-    }
-    // 如果最后没有新的控制量进入（没用新的期望状态输入），重复上一周期的最后一个控制量
-    nominal_tau[N-1] = prediction_tau[N-1];
-}
-
-// 使用U_bar和完整非线性ABA计算得到的状态序列X_bar作为预测状态序列
-// 从机器人当前真实状态开始，使用名义力矩U_bar，通过非线性ABA预测未来N步
-// 名义预测轨迹的起点，直接取机器人当前测量到的真实状态，然后再开始未来预测，进行N次循环，或则未来N次的状态序列X_bar
-void ROKAE_NMPC::generate_nom_traj(
-    const Eigen::VectorXd& current_state
-)
-{
-    // 将当前状态作为预测状态序列的第一个状态
-    nominal_state[0] = current_state;
-
-    // 使用ABA计算得到名义预测状态序列X_bar
-    for(int i = 0; i < N; ++i) {
-        nominal_state_next[i] = dynamics.compute_held_state(
-            nominal_state[i],
-            nominal_tau[i],
-            Ts
-        );
-        nominal_state[i+1] = nominal_state_next[i];
-    }
-}
-
-// 复用生成X_bar的函数，使用ABA计算得到预测状态序列X_bar，进行前向差分，这一步可以得到Mat_A[0]到Mat_A[19]和Mat_B[0]到Mat_B[19]
-// 也就是在名义状态附近线性化
-void ROKAE_NMPC::calculate_Mat_AB()
-{
-    dynamics.com_Mat_A_B(
-        nominal_state,
-        nominal_tau,
-        nominal_state_next,
-        Ts,
-        Mat_A,
-        Mat_B
-    );
-}
-
-// 总控制函数
-Eigen::VectorXd ROKAE_NMPC::compute_control(
-    const Eigen::VectorXd& current_state,
-    const std::vector<Eigen::VectorXd>& state_ref,
-    const std::vector<Eigen::VectorXd>& ddq_ref
-) {
-    last_qp_success = false;
-    if (current_state.size() != x_n || !current_state.allFinite() ||
-        state_ref.size() != static_cast<std::size_t>(N + 1) ||
-        ddq_ref.size() != static_cast<std::size_t>(N)) {
-        throw std::invalid_argument("Expected finite state (12), state_ref (N+1), ddq_ref (N).");
-    }
-    for (const auto& state : state_ref) {
-        if (state.size() != x_n || !state.allFinite()) {
-            throw std::invalid_argument("Each reference state must have 12 finite entries.");
-        }
-    }
-    for (const auto& acceleration : ddq_ref) {
-        if (acceleration.size() != u_n || !acceleration.allFinite()) {
-            throw std::invalid_argument("Each reference acceleration must have 6 finite entries.");
-        }
-    }
-
-    // 创建保存预测力矩的容器，20个6维控制向量
-    std::vector<Eigen::VectorXd> control_ref(N, Eigen::VectorXd::Zero(u_n));
-    // 使用RNEA计算参考力矩
-    // 如果参考轨迹能够完美实现，理论上的控制力矩大概是多少
-    for(int i = 0;i < N;i++){
-        control_ref[i] = dynamics.compute_rnea(
-            state_ref[i].head(DOF), // 关节位置
-            state_ref[i].tail(DOF), // 关节速度
-            ddq_ref[i] // 关节加速度
-        );
-    }
-
-    // 第一次使用RNEA计算得到的参考力矩初始化U_bar，后续周期使用上一个周期的最优控制序列向前移动一位
-    if (first_control_cycle) {
-        initial_nom_ctrl(control_ref);
-    } else {
-        shift_nom_ctrl();
-    }
-
-    // 20次ABA计算得到名义预测状态序列X_bar
-    generate_nom_traj(current_state);
-
-    // 前向差分计算A和B矩阵
-    calculate_Mat_AB();
-
-    // 构造Delta X = phi*delta_x0 +  gamma*delta_U
-    MPC_Matrices matrices = compute_mpc_matrices(
-        Mat_A,
-        Mat_B,
-        Q,
-        R,
-        F,
-        N
-    );
-
-    // 上面的状态可以拼成一个大向量
-    Eigen::VectorXd nominal_X(N*x_n);
-    Eigen::VectorXd nominal_U(N*u_n);
-    // 参考目标
-    Eigen::VectorXd ref_X(N*x_n);
-
-    // RNEA计算出来的参考力矩
-    Eigen::VectorXd ref_U(N*u_n);
-
-    // .segment(开始位置, 长度)
-    for(int i = 0; i<N;i++){
-        nominal_X.segment(i*x_n, x_n) = nominal_state[i+1];
-        nominal_U.segment(i*u_n, u_n) = nominal_tau[i];
-        ref_X.segment(i*x_n, x_n) = state_ref[i+1];
-        ref_U.segment(i*u_n, u_n) = control_ref[i];
-    }
-
-    // 因为名义轨迹从最新状态开始，所以delta_x0 = 0
-    Eigen::VectorXd delta_x0 = current_state - nominal_state[0];
-
-    // 构造代价函数并且求解QP，得到120个\Delta{U}^*
-    Eigen::VectorXd delta_U = Prediction(
-        delta_x0,
-        nominal_X,
-        nominal_U,
-        ref_X,
-        ref_U,
-        matrices,
-        tau_lower,
-        tau_upper,
-        N,
-        u_n,
-        &last_qp_success
-    );
-    // C++实机入口同样必须拒绝失败结果，不能把零增量当成有效控制量。
-    if (!last_qp_success || !delta_U.allFinite()) {
-        last_qp_success = false;
-        throw std::runtime_error("MPC QP failed; no control torque produced.");
-    }
-    first_control_cycle = false;
-
-    // U = U_bar + delta_U
-    for(int i = 0;i<N;i++)
-    {
-        delta_tau[i] = delta_U.segment(i*u_n,u_n);
-        prediction_tau[i] = nominal_tau[i] + delta_tau[i];
-    }
-
-    // X = X_bar +phi*delta_x0+gamma*delat U
-    Eigen::VectorXd delat_X = matrices.Phi*delta_x0+matrices.Gamma*delta_U;
-
-    prediction_state[0] = current_state;
-    for(int i = 0;i<N;i++)
-    {
-        prediction_state[i+1] = nominal_state[i+1] + delat_X.segment(i*x_n,x_n);
-    }
-
-    // 滚动时域控制之执行第一步
-    return prediction_tau[0];
-}
+} // namespace mpc_v2
```

<a id="file-python"></a>

## 6. python_bindings.cpp

源文件：[src/python_bindings.cpp](src/python_bindings.cpp) → [src_v2/python_bindings.cpp](src_v2/python_bindings.cpp)。

### 6.1 Python 调用方式变化

```python
# src 对应模块
from rokae_mpc import MPCController
controller = MPCController(urdf_path, timestep=0.001, horizon=40)

# src_v2 对应模块
from rokae_mpc_v2 import MPCController
controller = MPCController(
    urdf_path, timestep=0.001, horizon=40,
    gravity_compensated=True, integration_step=0.001, warm_start=True,
)
```

模块名是绑定源码中写定的名称，不是根据 `.cpp` 文件夹名称自动推断。以上示例只说明接口，本文没有导入或执行模块。

两版都接受 `state.shape == (12,)`、`state_ref.shape == (N+1,12)`、`ddq_ref.shape == (N,6)`，返回 `(6,)` 力矩。两版 `predict_state` 都返回 `(12,)` 状态，预测时的力矩保持不变。

### 6.2 逐项变化

| 项目 | src | src_v2 |
|---|---|---|
| 模块名 | `rokae_mpc` | `rokae_mpc_v2` |
| 内部对象 | `pinocchioFun`、`ROKAE_NMPC` | `Dynamics`、`Controller` |
| 构造参数 | URDF、timestep、horizon | 增加重力约定、小步长、QP 热启动开关 |
| 状态数据类型 | 动态 Eigen 向量 | 固定尺寸 State/Joint，绑定层负责类型及尺寸转换 |
| 参考数据转换 | 把 NumPy 对应矩阵逐行复制到 vector，再调用旧控制器 | 直接把矩阵传给 v2 控制器，省掉显式的 vector 拼装 |
| 校验位置 | 包装层手动检查形状、有限性和 QP 成功标志 | 主要交给固定尺寸转换和 v2 核心函数检查，包装函数较短 |
| `timestep`、`horizon` | 只读属性 | 保留 |
| `compute_control`、`predict_state` | 已提供 | 保留用途，内部改调用 v2 |
| `linearize` | 无 | 新增，返回 `(next_state, A, B)`，形状分别为 `(12,)`、`(12,12)`、`(12,6)` |
| `acceleration` | 无 | 新增，输入 state/tau，返回 `(6,)` 加速度 |
| `set_torque_limits` | 无 | 新增，接收两组 `(6,)` 上下界 |
| `reset` | 无 | 新增，重置控制器历史状态 |
| `timing` | 无 | 新增只读属性，返回 Timing 对象；七个统计字段也以只读属性暴露 |
| 对象互斥 | 无显式 mutex | 包装类持有 mutex，预测、求解、线性化、加速度、重置、限幅和统计读取均加锁 |
| GIL | 两个计算方法已释放 GIL | 六个计算/修改方法释放 GIL；timing 属性读取没有 call_guard |
| 新增依赖 | — | `pybind11/stl.h`、`mutex`、`tuple`，支持新包装实现与 tuple 返回 |

v2 的锁让同一个控制器对象上的这些操作串行执行，不是让一次 MPC 并行加速。`timestep` 和 `horizon` 不加锁，读取的是构造后不再修改的参数。

### 6.3 一个需要注意的线程细节

绑定中的 `timing` 属性会获取 C++ mutex，但读取属性时仍持有 Python GIL。如果另一个线程正在同一对象上做较长求解，读取 timing 会等待 mutex，这段等待可能阻塞其他 Python 线程。适合在求解结束后或执行求解的同一工作线程里读取，不能把“支持属性查询”理解为完全无阻塞监控。

### 6.4 完整逐行 diff

```diff
--- src/python_bindings.cpp
+++ src_v2/python_bindings.cpp
@@ -1,90 +1,90 @@
 #include "NMPC_control.h"
-#include "pinocchio_fun.h"
-
 #include <pybind11/eigen.h>
 #include <pybind11/pybind11.h>
-#include <stdexcept>
+#include <pybind11/stl.h>
+#include <mutex>
+#include <tuple>
 
-// 这样pybind11::init就等价于py::init
 namespace py = pybind11;
+using namespace mpc_v2;
 
-// 把动力学对象和控制器放在一起，保证 dynamics 的生命周期足够长。
+// Python 对象自己拥有 Dynamics，避免引用已析构对象。
+// 锁用于防止调用方误将同一个实例同时用于预测和求解，不是并行加速。
 class MPCController {
-    pinocchioFun dynamics;
-    ROKAE_NMPC controller;
+public:
+    MPCController(const std::string& urdf, double timestep, int horizon,
+                  bool gravity_compensated, double integration_step, bool warm_start)
+        : dynamics_(urdf, gravity_compensated, integration_step),
+          controller_(dynamics_, timestep, horizon, warm_start) {}
 
-public:
-    explicit MPCController(const std::string& urdf_path, double timestep, int horizon)
-        : dynamics(urdf_path), controller(dynamics, timestep, horizon) {}
-
-    double timestep() const 
-    { 
-        return controller.timestep();
+    Joint compute_control(const State& state, const Eigen::MatrixXd& refs,
+                          const Eigen::MatrixXd& accelerations) {
+        std::lock_guard<std::mutex> lock(mutex_);
+        return controller_.compute_control(state, refs, accelerations);
     }
-    int horizon() const
-    { 
-        return controller.horizon(); 
+    State predict_state(const State& state, const Joint& tau, double duration) {
+        std::lock_guard<std::mutex> lock(mutex_);
+        return dynamics_.integrate(state, tau, duration);
     }
-
-    // 这一段代码可以使得python代码也可以调用dynamics.compute_held_state()
-    Eigen::VectorXd predict_state(const Eigen::VectorXd& state,
-                                 const Eigen::VectorXd& tau, double duration)
-    {
-        if(state.size() != 12 || tau.size() != 6 || !state.allFinite() || !tau.allFinite()) {
-            throw std::invalid_argument("Expected finite state (12,) and torque (6,).");
-        }
-        return dynamics.compute_held_state(state, tau, duration);
+    std::tuple<State, MatA, MatB> linearize(const State& state, const Joint& tau, double duration) {
+        std::lock_guard<std::mutex> lock(mutex_);
+        const auto result = dynamics_.integrate_linearized(state, tau, duration);
+        return {result.next, result.A, result.B};
     }
-
-    // 关键函数,state表示当前真实机器人状态，state_ref是整个预测时域上的参考状态
-    Eigen::VectorXd compute_control(
-        const Eigen::VectorXd& state,
-        const Eigen::MatrixXd& state_ref,
-        const Eigen::MatrixXd& ddq_ref)
-    {
-        const int N = horizon();
-        // 尺寸检查， NumPy 的每一行对应一个预测时刻，避免错误输入导致 C++ 越界
-        if(state.size() != 12 || state_ref.rows() != N+1 ||
-           state_ref.cols() != 12 || ddq_ref.rows() != N || ddq_ref.cols() != 6) {
-            throw std::invalid_argument(
-                "Expected state (12,), state_ref (N+1, 12), ddq_ref (N, 6).");
-        }
-        // 检查三组输入都不能出现NaN/inf
-        if(!state.allFinite() || !state_ref.allFinite() || !ddq_ref.allFinite()) {
-            throw std::invalid_argument("Controller inputs must be finite.");
-        }
-
-        // 
-        std::vector<Eigen::VectorXd> states(N+1), accelerations(N);
-
-        for(int i = 0; i <= N; ++i) 
-            states[i] = state_ref.row(i).transpose();
-        for(int i = 0; i < N; ++i) 
-            accelerations[i] = ddq_ref.row(i).transpose();
-
-        Eigen::VectorXd tau = controller.compute_control(state, states, accelerations);
-        // 控制器失败时会抛出异常；绑定层再检查输出，避免将无效力矩用于仿真。
-        if(!controller.qp_success() || !tau.allFinite()) {
-            throw std::runtime_error("MPC QP failed; tracking experiment stopped.");
-        }
-        return tau;
+    Joint acceleration(const State& state, const Joint& tau) {
+        std::lock_guard<std::mutex> lock(mutex_);
+        return dynamics_.acceleration(state, tau);
     }
+    void reset() {
+        std::lock_guard<std::mutex> lock(mutex_);
+        controller_.reset();
+    }
+    void set_torque_limits(const Joint& lower, const Joint& upper) {
+        std::lock_guard<std::mutex> lock(mutex_);
+        controller_.set_torque_limits(lower, upper);
+    }
+    Timing timing() {
+        std::lock_guard<std::mutex> lock(mutex_);
+        return controller_.timing();
+    }
+    double timestep() const { return controller_.timestep(); }
+    int horizon() const { return controller_.horizon(); }
+private:
+    Dynamics dynamics_;
+    Controller controller_;
+    std::mutex mutex_;
 };
 
-// 创建一个python模块，名称为rokae_mpc，module为变量的名字
-PYBIND11_MODULE(rokae_mpc, module) {
-    // module.doc为Python的模块说明
-    module.doc() = "ROKAE SR4 C++ MPC controller";
-    // 把 C++ 的 MPCController 类注册进 rokae_mpc 模块，在 Python 里面也叫 MPCController
-    py::class_<MPCController>(module, "MPCController")
-        .def(py::init<const std::string&, double, int>(), py::arg("urdf_path"),
-             py::arg("timestep") = 0.001, py::arg("horizon") = 40)
+PYBIND11_MODULE(rokae_mpc_v2, m) {
+    m.doc() = "CPU MPC v2: analytical sensitivities and changing-Hessian QP warm start";
+    py::class_<Timing>(m, "Timing")
+        .def_readonly("reference_ms", &Timing::reference_ms)
+        .def_readonly("dynamics_ms", &Timing::dynamics_ms)
+        .def_readonly("matrices_ms", &Timing::matrices_ms)
+        .def_readonly("qp_ms", &Timing::qp_ms)
+        .def_readonly("total_ms", &Timing::total_ms)
+        .def_readonly("qp_iterations", &Timing::qp_iterations)
+        .def_readonly("qp_success", &Timing::qp_success);
+    py::class_<MPCController>(m, "MPCController")
+        .def(py::init<const std::string&, double, int, bool, double, bool>(),
+             py::arg("urdf_path"), py::arg("timestep") = 0.001, py::arg("horizon") = 40,
+             py::arg("gravity_compensated") = true, py::arg("integration_step") = 0.001,
+             py::arg("warm_start") = true)
         .def_property_readonly("timestep", &MPCController::timestep)
         .def_property_readonly("horizon", &MPCController::horizon)
+        .def_property_readonly("timing", &MPCController::timing)
+        .def("compute_control", &MPCController::compute_control,
+             py::arg("state"), py::arg("state_ref"), py::arg("ddq_ref"),
+             py::call_guard<py::gil_scoped_release>())
         .def("predict_state", &MPCController::predict_state,
              py::arg("state"), py::arg("tau"), py::arg("duration"),
              py::call_guard<py::gil_scoped_release>())
-        .def("compute_control", &MPCController::compute_control,
-             py::arg("state"), py::arg("state_ref"), py::arg("ddq_ref"),
-             py::call_guard<py::gil_scoped_release>());
+        .def("linearize", &MPCController::linearize,
+             py::arg("state"), py::arg("tau"), py::arg("duration"),
+             py::call_guard<py::gil_scoped_release>())
+        .def("acceleration", &MPCController::acceleration,
+             py::arg("state"), py::arg("tau"), py::call_guard<py::gil_scoped_release>())
+        .def("set_torque_limits", &MPCController::set_torque_limits,
+             py::arg("lower"), py::arg("upper"), py::call_guard<py::gil_scoped_release>())
+        .def("reset", &MPCController::reset, py::call_guard<py::gil_scoped_release>());
 }
```

<a id="file-main"></a>

## 7. main.cpp：仅 src 中存在

源文件：[src/main.cpp](src/main.cpp)。`src_v2/main.cpp` 不存在，因此目录对照中显示为仅旧目录存在；这不表示本次任务删除了原文件。

这个文件提供一个独立 C++ 仿真入口：

- 第一个命令行参数指定 URDF，第二个指定输出文件；默认分别为 `Robot_NMPC/urdf/ROKAE_SR4.urdf` 和 `step_response.txt`。
- 创建旧版动力学和控制器，并设置差分步长为 `1e-6、1e-6、1e-4`。
- 从控制器读取周期与预测步数，六个关节从零位置、零速度开始。
- 关节 1 的参考位置为 30°，其余参考位置、速度和加速度为零。
- 使用同一个 Pinocchio 动力学模型模拟约 3 秒运动；默认周期 1 ms 时是 3000 步。
- 每次计算力矩后，用 `compute_held_state` 推进虚拟状态。
- 输出时间、关节 1 参考角度、六关节实际角度和六关节力矩，约每 0.2 秒打印一次进度。
- 捕获异常并返回 1，正常结束返回 0。

它不是 MuJoCo 入口，也不包含机器人 SDK 通信。v2 目录没有同名入口，不代表 v2 没有控制器；v2 核心和 Python 绑定仍然存在。是否生成某个可执行文件还取决于构建配置，本次不对 CMake 作差异审计。

### 7.1 完整逐行 diff

```diff
--- src/main.cpp
+++ /dev/null
@@ -1,132 +0,0 @@
-#include "NMPC_control.h"
-#include "pinocchio_fun.h"
-
-#include <Eigen/Dense>
-#include <cmath>
-#include <fstream>
-#include <iomanip>
-#include <iostream>
-#include <stdexcept>
-#include <string>
-#include <vector>
-
-int main(int argc, char* argv[])
-{
-    const double pi = 3.14159265358979323846;
-    const double rad_to_deg = 180.0 / pi;
-    const double step_angle_deg = 30.0;    // 关节1阶跃角度
-
-    // 第一个参数可以指定URDF，第二个参数可以指定输出文件
-    std::string urdf_filename = (argc >= 2)
-        ? argv[1]
-        : "Robot_NMPC/urdf/ROKAE_SR4.urdf";
-    std::string output_filename = (argc >= 3)
-        ? argv[2]
-        : "step_response.txt";
-
-    try {
-        // 1. 创建SR4动力学模型和LTV-MPC控制器
-        pinocchioFun robot_dynamics(urdf_filename);
-        robot_dynamics.set_forward_diff_step(1e-6, 1e-6, 1e-4);
-
-        ROKAE_NMPC controller(robot_dynamics);
-        // 从控制器读取参数，避免仿真周期与 MPC 周期不一致。
-        const double Ts = controller.timestep();
-        const int prediction_horizon = controller.horizon();
-        const int simulation_steps = static_cast<int>(std::lround(3.0 / Ts));
-        const int print_interval = static_cast<int>(std::lround(0.2 / Ts));
-
-        // 2. 初始状态：六个关节的位置和速度全部为零
-        Eigen::VectorXd current_state = Eigen::VectorXd::Zero(12);
-
-        // 3. 给关节1施加从0度到30度的阶跃参考，其他关节参考保持0度
-        Eigen::VectorXd q_ref = Eigen::VectorXd::Zero(6);
-        q_ref(0) = step_angle_deg / rad_to_deg;
-
-        Eigen::VectorXd ref_state = Eigen::VectorXd::Zero(12);
-        ref_state.head(6) = q_ref;
-
-        std::vector<Eigen::VectorXd> state_reference(
-            prediction_horizon + 1,
-            ref_state);
-        std::vector<Eigen::VectorXd> acceleration_reference(
-            prediction_horizon,
-            Eigen::VectorXd::Zero(6));
-
-        // 4. 创建阶跃响应输出文件
-        std::ofstream output_file(output_filename);
-        if(!output_file.is_open()) {
-            throw std::runtime_error("无法创建阶跃响应输出文件.");
-        }
-
-        output_file << std::fixed << std::setprecision(10);
-        output_file
-            << "time_s q1_ref_deg "
-            << "q1_deg q2_deg q3_deg q4_deg q5_deg q6_deg "
-            << "tau1_Nm tau2_Nm tau3_Nm tau4_Nm tau5_Nm tau6_Nm"
-            << std::endl;
-
-        // 记录t=0时刻：关节位置为零，此时还没有施加控制力矩
-        output_file << 0.0 << " " << step_angle_deg;
-        for(int joint = 0; joint < 6; joint++) {
-            output_file << " " << current_state(joint) * rad_to_deg;
-        }
-        for(int joint = 0; joint < 6; joint++) {
-            output_file << " " << 0.0;
-        }
-        output_file << std::endl;
-
-        std::cout << "Use robot model: " << urdf_filename << std::endl;
-        std::cout << "Joint 1 step reference: "
-                  << step_angle_deg << " deg" << std::endl;
-        std::cout << "Control frequency: " << 1.0 / Ts << " Hz" << std::endl;
-
-        // 5. 闭环运行3秒；1000 Hz时共有3000个控制周期
-        for(int step = 0; step < simulation_steps; step++) {
-            // MPC输入当前状态，输出当前周期应该执行的六维关节力矩
-            Eigen::VectorXd control = controller.compute_control(
-                current_state,
-                state_reference,
-                acceleration_reference);
-
-            // 使用与控制器相同的周期模拟机器人向前运动一步
-            current_state = robot_dynamics.compute_held_state(
-                current_state,
-                control,
-                Ts);
-
-            double current_time = (step + 1) * Ts;
-
-            // 每一行保存：时间、关节1参考角度、六关节实际角度、六关节力矩
-            output_file << current_time << " " << step_angle_deg;
-            for(int joint = 0; joint < 6; joint++) {
-                output_file << " " << current_state(joint) * rad_to_deg;
-            }
-            for(int joint = 0; joint < 6; joint++) {
-                output_file << " " << control(joint);
-            }
-            output_file << std::endl;
-
-            // 每0.2秒在终端显示一次关节1的响应
-            if((step + 1) % print_interval == 0) {
-                std::cout << "time = " << current_time
-                          << " s, q1 = "
-                          << current_state(0) * rad_to_deg
-                          << " deg, tau1 = "
-                          << control(0)
-                          << " Nm"
-                          << std::endl;
-            }
-        }
-
-        output_file.close();
-        std::cout << "Step response saved to: "
-                  << output_filename << std::endl;
-    }
-    catch(const std::exception& error) {
-        std::cerr << "LTV-MPC运行失败：" << error.what() << std::endl;
-        return 1;
-    }
-
-    return 0;
-}
```

## 8. 对迁移和结果比较的影响

| 问题 | 根据当前源码能够得出的结论 |
|---|---|
| 是否只是改了类名？ | 不是。导数计算、矩阵组织、QP 状态保存和失败恢复都有变化。 |
| 输入与输出的物理意义是否改变？ | 默认没有：q 为 rad，dq 为 rad/s，ddq 为 rad/s²，输出为不含重力项的附加力矩 N·m。 |
| 是否还能使用原来的 C++ 调用代码？ | 不能只换 include 路径；类名和参考轨迹容器类型也需要修改。 |
| Python 是否容易迁移？ | 核心 compute_control/predict_state 的数组形状保持一致，但需要使用 `rokae_mpc_v2`，新增选项按需要设置。 |
| v2 是否对所有设置与旧版完全等价？ | 不是。导数计算不同、QP 热启动不同，浮点误差与数值路径不同；v2 矩阵类还固定了对角权重结构。 |
| v2 是否取消了 ABA？ | 没有。仍计算名义动力学和解析导数，取消的是逐维差分所需的重复扰动积分。 |
| v2 是否保证没有任何内存分配？ | 不能这样说。代码复用了主要缓存，但矩阵表达式、库内部和绑定转换仍可能分配。 |
| 能否单凭代码确认达到 50 Hz？ | 不能。需要在目标机器上测量总耗时及超时情况；本次没有性能测试。 |
| gravity_compensated 应如何设置？ | 对当前底层始终补偿重力的机器人，保持 true；设为 false 会恢复模型重力项。 |

为了比较控制效果，应保持相同 URDF、参考轨迹、采样周期、预测步数、积分小步长、重力约定和力矩限制。解析导数与差分产生的细小数值差异并不自动说明哪一版实现错误，需要结合误差、约束和求解状态判断。

## 9. 本次文件快照

下表的 SHA-256 对应原始文件字节，用于辨认本文针对的具体版本。以后修改源码后，需要重新生成 diff；本文不是自动随代码更新的页面。

| 文件 | SHA-256 |
|---|---|
| `src/MPC_Matrices.cpp` | `b46e9f9e9f2d0f82ddff0319d6797485083b9c555225c9e50c902374278e45e1` |
| `src_v2/MPC_Matrices.cpp` | `a43b3646c766c05b053570a6053518d941f182157f480d49f62147a02ef09188` |
| `src/NMPC_control.cpp` | `27492ed3f71427533cdb7d9a25a8102cc5ae7519c820fdb57701ec2f4809fc9f` |
| `src_v2/NMPC_control.cpp` | `f0912011303b4aa16044031b22c68733ac7beff28415451287a5fea1ea3a2221` |
| `src/Prediction.cpp` | `baf41cd1981acd3c7106a42cc19080db3ad7ad1d7331250377913c5640909d95` |
| `src_v2/Prediction.cpp` | `1027d24cb531e3087dc10631cfa0b7addbfc71aa9456c2d9ffb298a9039eeaa8` |
| `src/main.cpp` | `ca5bf30509be39e5a14073092182144ae262d71d3e855719107ad19fbd55f034` |
| `src/pinocchio_fun.cpp` | `5913cfe17be71eb2daa13bce02712456244c8d1a3d2276632f27927e4b2239e6` |
| `src_v2/pinocchio_fun.cpp` | `04f402abde20955489d7a671b078479e50c690dff3bf383ffc381c84fcc483a2` |
| `src/python_bindings.cpp` | `3c12450b9ec147f7ab7bf2b9e77b13066b39700bdb10512f75896d744e88f1ab` |
| `src_v2/python_bindings.cpp` | `78e70f7bede637af2fd94dc3ff8febd1ab85c362008a60dacc627f284e4690b2` |
