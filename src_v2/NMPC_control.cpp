#include "NMPC_control.h"
#include <algorithm>
#include <chrono>

namespace mpc_v2 {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end-start).count();
}
}

Controller::Controller(Dynamics& dynamics, double timestep, int horizon, bool warm_start)
    : dynamics_(dynamics), N_(checked_horizon(horizon)), Ts_(timestep),
      x_(N_+1), u_(N_), optimal_u_(N_), A_(N_), B_(N_),
      nominal_u_(N_*DOF), reference_u_(N_*DOF), state_error_(N_*NX),
      matrices_(N_), qp_(N_, warm_start) {
    require(std::isfinite(Ts_) && Ts_ > 0, "timestep must be positive");
}

void Controller::reset() {
    first_ = true;
    qp_.reset();
    timing_ = Timing{};
}

void Controller::set_torque_limits(const Joint& lower, const Joint& upper) {
    require(lower.allFinite() && upper.allFinite() && (lower.array() < upper.array()).all(),
            "Expected finite lower < upper torque limits");
    lower_ = lower;
    upper_ = upper;
    reset();
}

Joint Controller::compute_control(const State& state, const Eigen::MatrixXd& state_ref,
                                 const Eigen::MatrixXd& ddq_ref) {
    const auto start = Clock::now();
    timing_ = Timing{};
    try {
        require(state_ref.rows() == N_+1 && state_ref.cols() == NX &&
                ddq_ref.rows() == N_ && ddq_ref.cols() == DOF, "Reference shape mismatch");
        require(state.allFinite() && state_ref.allFinite() && ddq_ref.allFinite(),
                "Nonfinite controller input");

        // 1. 与旧版一样，用参考轨迹的 RNEA 生成参考力矩。
        for (int i = 0; i < N_; ++i) {
            const State ref = state_ref.row(i).transpose();
            const Joint ddq = ddq_ref.row(i).transpose();
            reference_u_.segment<DOF>(i*DOF) = dynamics_.inverse_dynamics(ref, ddq);
            u_[i] = first_ ? Joint(reference_u_.segment<DOF>(i*DOF))
                          : optimal_u_[std::min(i+1, N_-1)];
            nominal_u_.segment<DOF>(i*DOF) = u_[i];
        }
        const auto after_reference = Clock::now();
        timing_.reference_ms = milliseconds(start, after_reference);

        // 2. 只积分一条名义轨迹，同时得到每个区间的 A、B。
        x_[0] = state;
        for (int i = 0; i < N_; ++i) {
            const auto step = dynamics_.integrate_linearized(x_[i], u_[i], Ts_);
            x_[i+1] = step.next;
            A_[i] = step.A;
            B_[i] = step.B;
            state_error_.segment<NX>(i*NX) = x_[i+1]-state_ref.row(i+1).transpose();
        }
        const auto after_dynamics = Clock::now();
        timing_.dynamics_ms = milliseconds(after_reference, after_dynamics);

        // 3. delta_x0=0，所以 delta_X = Gamma * delta_U。
        matrices_.update(A_, B_);
        qp_.prepare(matrices_, state_error_, nominal_u_, reference_u_, lower_, upper_);
        const auto after_matrices = Clock::now();
        timing_.matrices_ms = milliseconds(after_dynamics, after_matrices);

        // 4. 求解器持续保存，H 变化时也能正确热启动。
        const Eigen::VectorXd& delta_u = qp_.solve();
        timing_.qp_ms = milliseconds(after_matrices, Clock::now());
        timing_.qp_iterations = qp_.iterations();
        for (int i = 0; i < N_; ++i) {
            optimal_u_[i] = u_[i] + delta_u.segment<DOF>(i*DOF);
            if (!optimal_u_[i].allFinite() || (optimal_u_[i]-lower_).minCoeff() < -1e-5 ||
                (upper_-optimal_u_[i]).minCoeff() < -1e-5)
                throw std::runtime_error("Invalid final torque sequence");
        }
        first_ = false;
        timing_.qp_success = true;
        timing_.total_ms = milliseconds(start, Clock::now());
        return optimal_u_[0];
    } catch (...) {
        // 下次从参考重新初始化；当前必须把失败交给调用方处理。
        first_ = true;
        qp_.reset();
        timing_.qp_success = false;
        timing_.total_ms = milliseconds(start, Clock::now());
        throw;
    }
}

} // namespace mpc_v2
