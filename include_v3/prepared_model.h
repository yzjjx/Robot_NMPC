#pragma once
#include "dynamics_backend.h"
#include "prepared_qp.h"

namespace mpc_v3 {
struct PhaseModel {
    State x0;
    double first_duration = 0;
    Eigen::MatrixXd nominal_x, nominal_u;
    Eigen::MatrixXd Phi, Gamma;
    Eigen::VectorXd eta;
    std::unique_ptr<PreparedQP> qp;
};
struct PreparedBlock {
    std::int64_t index = 0;
    std::vector<std::shared_ptr<PhaseModel>> phases;
    PreparationInfo info;
};

// 测试也会使用这个递推，防止遗漏动力学缺陷。
void condense(const std::vector<MatA>& A, const std::vector<MatB>& B,
              const std::vector<State>& defects, Eigen::MatrixXd& Phi,
              Eigen::MatrixXd& Gamma, Eigen::VectorXd& eta);

std::shared_ptr<PreparedBlock> prepare_block(DynamicsBackend& backend, const Config& config,
    std::int64_t block, const Eigen::MatrixXd& nominal_x, const Eigen::MatrixXd& nominal_u,
    const Eigen::MatrixXd& reference_x, const Eigen::MatrixXd& reference_u);
} // namespace mpc_v3
