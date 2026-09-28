#pragma once

#include "MPC_Matrices.h"
#include <qpOASES.hpp>
#include <array>

namespace mpc_v2 {

class Prediction {
public:
    explicit Prediction(int horizon, bool warm_start = true);
    Prediction(const Prediction&) = delete;
    Prediction& operator=(const Prediction&) = delete;
    void prepare(const MPCMatrices& matrices, const Eigen::VectorXd& state_error,
                 const Eigen::VectorXd& nominal_u, const Eigen::VectorXd& reference_u,
                 const Joint& lower, const Joint& upper);
    const Eigen::VectorXd& solve();
    void reset();
    int iterations() const { return iterations_; }

private:
    using QpMatrix = Eigen::Matrix<qpOASES::real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using QpVector = Eigen::Matrix<qpOASES::real_t, Eigen::Dynamic, 1>;
    int n_;
    // qpOASES 对 Hessian 使用浅拷贝。双份 H 保证写新 H 时旧 H 仍有效。
    std::array<QpMatrix, 2> H_;
    QpVector g_, lower_, upper_, solution_;
    Eigen::VectorXd weighted_error_, delta_;
    qpOASES::SQProblem solver_; // 声明在 H 后面，析构时先释放 solver。
    bool warm_start_, initialized_ = false, prepared_ = false;
    int active_ = -1, pending_ = 0, iterations_ = 0;
};

} // namespace mpc_v2
