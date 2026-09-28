#include "MPC_Matrices.h"

namespace mpc_v2 {

MPCMatrices::MPCMatrices(int horizon) : N_(checked_horizon(horizon)) {
    Gamma.setZero(N_*NX, N_*DOF);
    weighted_Gamma.resize(N_*NX, N_*DOF);
    row_.setZero(NX, N_*DOF);
    next_row_.resize(NX, N_*DOF);
    state_weights.resize(N_*NX);
    control_weights.setConstant(N_*DOF, 0.01);
    // 与旧版一致：Q=diag(100,...,100,1,...,1)，终端 F=2Q。
    for (int i = 0; i < N_; ++i) {
        const double terminal = (i == N_-1) ? 2.0 : 1.0;
        state_weights.segment(i*NX, DOF).setConstant(100.0*terminal);
        state_weights.segment(i*NX+DOF, DOF).setConstant(terminal);
    }
}

void MPCMatrices::update(const std::vector<MatA>& A, const std::vector<MatB>& B) {
    require(static_cast<int>(A.size()) == N_ && static_cast<int>(B.size()) == N_,
            "A/B horizon mismatch");
    row_.setZero();
    for (int i = 0; i < N_; ++i) {
        next_row_.noalias() = A[i]*row_;
        next_row_.block(0, i*DOF, NX, DOF) += B[i];
        row_.swap(next_row_);
        Gamma.middleRows(i*NX, NX) = row_;
    }
    // 对角权重直接乘每一行，不构造巨大的 Q_bar。
    weighted_Gamma = Gamma;
    for (int r = 0; r < Gamma.rows(); ++r)
        weighted_Gamma.row(r) *= state_weights(r);
}

} // namespace mpc_v2
