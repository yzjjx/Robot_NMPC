#include "Prediction.h"
#include <string>

namespace mpc_v2 {

Prediction::Prediction(
    int horizon,
    bool warm_start
)
    : nV(checked_horizon(horizon) * DOF),
      qp_solver(nV, 0),
      use_warm_start(warm_start)
{
    for (auto& H_qp : H_buffer) {
        H_qp.resize(nV, nV);
    }
    g_qp.resize(nV);
    lb.resize(nV);
    ub.resize(nV);
    solution.resize(nV);
    weighted_state_error.resize(horizon * NX);
    delta_U.resize(nV);
    qpOASES::Options options;
    options.printLevel = qpOASES::PL_NONE;
    qp_solver.setOptions(options);
}

void Prediction::reset()
{
    qp_solver.reset();
    solver_initialized = false;
    qp_prepared = false;
    active_hessian_index = -1;
}

void Prediction::prepare(
    const MPCMatrices& mpc_matrices,
    const Eigen::VectorXd& state_error,
    const Eigen::VectorXd& nominal_U,
    const Eigen::VectorXd& ref_U,
    const Joint& tau_lower,
    const Joint& tau_upper
)
{
    require(
        state_error.size() == mpc_matrices.Gamma.rows() && nominal_U.size() == nV && ref_U.size() == nV,
        "QP dimension mismatch"
    );
    qp_prepared = false;
    pending_hessian_index = (active_hessian_index == 0) ? 1 : 0;
    auto& H_qp = H_buffer[pending_hessian_index];
    // 与旧版相同的目标函数、正则化；不降低精度、不放松力矩上下界。
    H_qp = (2.0 * mpc_matrices.Gamma.transpose() * mpc_matrices.weighted_Gamma).cast<qpOASES::real_t>();
    for (int i = 0; i < nV; ++i) {
        H_qp(i, i) += static_cast<qpOASES::real_t>(2.0 * mpc_matrices.control_weights(i) + 1e-6);
    }
    weighted_state_error = mpc_matrices.state_weights.cwiseProduct(state_error);
    delta_U.noalias() = mpc_matrices.Gamma.transpose() * weighted_state_error;
    delta_U += mpc_matrices.control_weights.cwiseProduct(nominal_U - ref_U);
    g_qp = (2.0 * delta_U).cast<qpOASES::real_t>();
    for (int i = 0; i < nV; ++i) {
        lb(i) = static_cast<qpOASES::real_t>(tau_lower(i % DOF) - nominal_U(i));
        ub(i) = static_cast<qpOASES::real_t>(tau_upper(i % DOF) - nominal_U(i));
    }
    if (!H_qp.allFinite() || !g_qp.allFinite() || !lb.allFinite() || !ub.allFinite()) {
        throw std::runtime_error("Nonfinite QP data");
    }
    qp_prepared = true;
}

const Eigen::VectorXd& Prediction::solve()
{
    require(qp_prepared, "Call prepare before solve");
    qp_prepared = false;
    qp_iterations = 500; // 上限，不是每次必须迭代 500 次；不宣称硬实时。
    qpOASES::returnValue status;
    if (solver_initialized && use_warm_start) {
        status = qp_solver.hotstart(
            H_buffer[pending_hessian_index].data(),
            g_qp.data(),
            static_cast<const qpOASES::real_t*>(nullptr),
            lb.data(),
            ub.data(),
            nullptr,
            nullptr,
            qp_iterations
        );
    } else {
        qp_solver.reset();
        status = qp_solver.init(
            H_buffer[pending_hessian_index].data(),
            g_qp.data(),
            static_cast<const qpOASES::real_t*>(nullptr),
            lb.data(),
            ub.data(),
            nullptr,
            nullptr,
            qp_iterations
        );
    }
    if (status == qpOASES::SUCCESSFUL_RETURN) {
        status = qp_solver.getPrimalSolution(solution.data());
    }
    if (status != qpOASES::SUCCESSFUL_RETURN || !solution.allFinite()) {
        reset();
        throw std::runtime_error("QP failed, qpOASES code=" + std::to_string(static_cast<int>(status)));
    }
    delta_U = solution.cast<double>();
    // 失败时不把零向量/上一周期结果伪装成有效解。
    const double tolerance = 1e-5;
    if ((solution - lb).minCoeff() < -tolerance || (ub - solution).minCoeff() < -tolerance) {
        reset();
        throw std::runtime_error("QP torque bounds violated");
    }
    solver_initialized = true;
    active_hessian_index = pending_hessian_index;
    return delta_U;
}

} // namespace mpc_v2
