#include "prepared_qp.h"
#include <string>
namespace mpc_v3 {
PreparedQP::PreparedQP(const Eigen::MatrixXd& H, const Eigen::VectorXd& g,
    const Eigen::MatrixXd& K, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub, int limit)
    : H_(H.cast<qpOASES::real_t>()), base_g_(g.cast<qpOASES::real_t>()),
      g_(base_g_), lb_(lb.cast<qpOASES::real_t>()), ub_(ub.cast<qpOASES::real_t>()),
      solution_(g.size()), K_(K), solver_(static_cast<int>(g.size())), max_iterations_(limit) {
    require(H.rows()==g.size() && H.cols()==g.size() && K.rows()==g.size() && K.cols()==NX &&
            lb.size()==g.size() && ub.size()==g.size(), "QP shape mismatch");
    require(H.allFinite() && g.allFinite() && K.allFinite() && lb.allFinite() && ub.allFinite() &&
            (lb.array()<=ub.array()).all(), "Invalid QP data");
    qpOASES::Options options;
    options.printLevel=qpOASES::PL_NONE;
    solver_.setOptions(options);
    iterations_=max_iterations_;
    // 冷启动在准备阶段完成，反馈只更新梯度；准备阶段也有实在的计算成本。
    const auto status=solver_.init(H_.data(),g_.data(),lb_.data(),ub_.data(),iterations_);
    if(status!=qpOASES::SUCCESSFUL_RETURN)
        throw std::runtime_error("Preparation QP failed: "+std::to_string(static_cast<int>(status)));
    valid_=true;
}

Eigen::VectorXd PreparedQP::solve(const State& dx) {
    if(!valid_) throw std::runtime_error("QP invalid; prepare a new model");
    require(dx.allFinite(), "Nonfinite state correction");
    g_=base_g_+(K_*dx).cast<qpOASES::real_t>();
    iterations_=max_iterations_;
    auto status=solver_.hotstart(g_.data(),lb_.data(),ub_.data(),iterations_);
    if(status==qpOASES::SUCCESSFUL_RETURN) status=solver_.getPrimalSolution(solution_.data());
    if(status!=qpOASES::SUCCESSFUL_RETURN || !solution_.allFinite() ||
       (solution_-lb_).minCoeff() < -1e-5 || (ub_-solution_).minCoeff() < -1e-5) {
        valid_=false;
        throw std::runtime_error("Feedback QP failed: "+std::to_string(static_cast<int>(status)));
    }
    return solution_.cast<double>();
}
} // namespace mpc_v3
