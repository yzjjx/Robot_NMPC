#include "MPC_Matrices.h"

namespace mpc_v2 {

MPCMatrices::MPCMatrices(
    int horizon
)
    : N(checked_horizon(horizon))
{
    Gamma.setZero(N * NX, N * DOF);
    weighted_Gamma.resize(N * NX, N * DOF);
    Gamma_i.setZero(NX, N * DOF);
    Gamma_next.resize(NX, N * DOF);
    state_weights.resize(N * NX);
    control_weights.setConstant(N * DOF, 0.01);
    // 与旧版一致：Q=diag(100,...,100,1,...,1)，终端 F=2Q。
    for (int i = 0; i < N; ++i) {
        const double terminal_scale = (i == N - 1) ? 2.0 : 1.0;
        state_weights.segment(i * NX, DOF).setConstant(100.0 * terminal_scale);
        state_weights.segment(i * NX + DOF, DOF).setConstant(terminal_scale);
    }
}

void MPCMatrices::update(
    const std::vector<MatA>& Mat_A,
    const std::vector<MatB>& Mat_B
)
{
    require(static_cast<int>(Mat_A.size()) == N && static_cast<int>(Mat_B.size()) == N, "A/B horizon mismatch");
    Gamma_i.setZero();
    for (int i = 0; i < N; ++i) {
        Gamma_next.noalias() = Mat_A[i] * Gamma_i;
        Gamma_next.block(0, i * DOF, NX, DOF) += Mat_B[i];
        Gamma_i.swap(Gamma_next);
        Gamma.middleRows(i * NX, NX) = Gamma_i;
    }
    // 对角权重直接乘每一行，不构造巨大的 Q_bar。
    weighted_Gamma = Gamma;
    for (int row = 0; row < Gamma.rows(); ++row) {
        weighted_Gamma.row(row) *= state_weights(row);
    }
}

} // namespace mpc_v2
