#include "Prediction.h"
#include <string>

namespace mpc_v2 {

Prediction::Prediction(int horizon, bool warm_start)
    : n_(checked_horizon(horizon)*DOF), solver_(n_, 0), warm_start_(warm_start) {
    for (auto& h : H_) h.resize(n_, n_);
    g_.resize(n_);
    lower_.resize(n_);
    upper_.resize(n_);
    solution_.resize(n_);
    weighted_error_.resize(horizon*NX);
    delta_.resize(n_);
    qpOASES::Options options;
    options.printLevel = qpOASES::PL_NONE;
    solver_.setOptions(options);
}

void Prediction::reset() {
    solver_.reset();
    initialized_ = false;
    prepared_ = false;
    active_ = -1;
}

void Prediction::prepare(const MPCMatrices& m, const Eigen::VectorXd& error,
                         const Eigen::VectorXd& nominal_u, const Eigen::VectorXd& reference_u,
                         const Joint& lower, const Joint& upper) {
    require(error.size() == m.Gamma.rows() && nominal_u.size() == n_ && reference_u.size() == n_,
            "QP dimension mismatch");
    prepared_ = false;
    pending_ = (active_ == 0) ? 1 : 0;
    auto& h = H_[pending_];
    // 与旧版相同的目标函数、正则化；不降低精度、不放松力矩上下界。
    h = (2.0*m.Gamma.transpose()*m.weighted_Gamma).cast<qpOASES::real_t>();
    for (int i = 0; i < n_; ++i)
        h(i, i) += static_cast<qpOASES::real_t>(2.0*m.control_weights(i) + 1e-6);
    weighted_error_ = m.state_weights.cwiseProduct(error);
    delta_.noalias() = m.Gamma.transpose()*weighted_error_;
    delta_ += m.control_weights.cwiseProduct(nominal_u-reference_u);
    g_ = (2.0*delta_).cast<qpOASES::real_t>();
    for (int i = 0; i < n_; ++i) {
        lower_(i) = static_cast<qpOASES::real_t>(lower(i % DOF)-nominal_u(i));
        upper_(i) = static_cast<qpOASES::real_t>(upper(i % DOF)-nominal_u(i));
    }
    if (!h.allFinite() || !g_.allFinite() || !lower_.allFinite() || !upper_.allFinite())
        throw std::runtime_error("Nonfinite QP data");
    prepared_ = true;
}

const Eigen::VectorXd& Prediction::solve() {
    require(prepared_, "Call prepare before solve");
    prepared_ = false;
    iterations_ = 500; // 上限，不是每次必须迭代 500 次；不宣称硬实时。
    qpOASES::returnValue status;
    if (initialized_ && warm_start_) {
        status = solver_.hotstart(H_[pending_].data(), g_.data(),
                                 static_cast<const qpOASES::real_t*>(nullptr),
                                 lower_.data(), upper_.data(), nullptr, nullptr, iterations_);
    } else {
        solver_.reset();
        status = solver_.init(H_[pending_].data(), g_.data(),
                              static_cast<const qpOASES::real_t*>(nullptr),
                              lower_.data(), upper_.data(), nullptr, nullptr, iterations_);
    }
    if (status == qpOASES::SUCCESSFUL_RETURN)
        status = solver_.getPrimalSolution(solution_.data());
    if (status != qpOASES::SUCCESSFUL_RETURN || !solution_.allFinite()) {
        reset();
        throw std::runtime_error("QP failed, qpOASES code=" + std::to_string(static_cast<int>(status)));
    }
    delta_ = solution_.cast<double>();
    // 失败时不把零向量/上一周期结果伪装成有效解。
    const double tolerance = 1e-5;
    if ((solution_-lower_).minCoeff() < -tolerance ||
        (upper_-solution_).minCoeff() < -tolerance) {
        reset();
        throw std::runtime_error("QP torque bounds violated");
    }
    initialized_ = true;
    active_ = pending_;
    return delta_;
}

} // namespace mpc_v2
