#include "MPC_Matrices.h"

#include <stdexcept>

// 接收已经算好的 N 个 A、B 矩阵，构造整个预测时域的矩阵。
namespace mpc_v2 {

// 构造函数只执行一次：分配矩阵空间，设置状态和控制权重。
MPCMatrices::MPCMatrices(
    int horizon
)
    : N(checked_horizon(horizon))
{
    const int n = NX;  // 状态维度12：6个位置、6个速度
    const int p = DOF; // 控制输入维度6：6个关节力矩

    // 原式：Delta X = Phi * delta_x0 + Gamma * Delta U。
    // 名义起点就是当前测量状态，delta_x0 = 0，因此不需要构造 Phi。
    Gamma.setZero(N*n, N*p);
    weighted_Gamma.resize(N*n, N*p);

    // Gamma_i 保存当前递推结果，Gamma_next 保存下一步结果。
    Gamma_i.setZero(n, N*p);
    Gamma_next.resize(n, N*p);

    // 状态权重：前 N-1 个节点使用 Q，最后一个节点使用 F。
    // 与 v1 相同，但只保存 Q_bar 的对角元素，不创建完整矩阵。
    state_weights.resize(N*n);
    for(int i = 0; i < N-1; i++) {
        state_weights.segment(i*n, p).setConstant(100.0); // Q：位置权重
        state_weights.segment(i*n+p, p).setConstant(1.0); // Q：速度权重
    }
    state_weights.segment((N-1)*n, p).setConstant(200.0); // F = 2Q
    state_weights.segment((N-1)*n+p, p).setConstant(2.0);

    // 控制权重：R = 0.01 * I，只保存 R_bar 的对角元素。
    control_weights.setConstant(N*p, 0.01);
}

// 每次 MPC 求解时调用：使用本次的 A、B 重新计算 Gamma。
void MPCMatrices::update(
    const std::vector<MatA>& Mat_A,
    const std::vector<MatB>& Mat_B
)
{
    if (static_cast<int>(Mat_A.size()) != N || static_cast<int>(Mat_B.size()) != N) {
        throw std::invalid_argument("A/B horizon mismatch");
    }

    const int n = NX;  // 状态维度
    const int p = DOF; // 控制输入维度

    // Delta X = Gamma * Delta U，每次从零开始递推。
    Gamma_i.setZero();
    for(int i = 0; i < N; i++) {
        // delta_x_{i+1} = A_i * delta_x_i + B_i * delta_u_i。
        // 先传播前面控制量的影响，再加入本节点控制量的影响。
        // noalias() 表示结果写入独立缓存，不会覆盖右侧正在使用的数据。
        Gamma_next.noalias() = Mat_A[i] * Gamma_i;
        Gamma_next.block(0, i*p, n, p) += Mat_B[i];

        // 交换缓存，让 Gamma_i 持有最新结果，避免再复制一次矩阵。
        Gamma_i.swap(Gamma_next);
        Gamma.block(i*n, 0, n, N*p) = Gamma_i;
    }

    // weighted_Gamma = Q_bar * Gamma。
    // Q_bar 是对角矩阵，左乘它等于将 Gamma 的每行乘以对应权重。
    weighted_Gamma = Gamma;
    for(int row = 0; row < Gamma.rows(); row++) {
        weighted_Gamma.row(row) *= state_weights(row);
    }
}

} // namespace mpc_v2
