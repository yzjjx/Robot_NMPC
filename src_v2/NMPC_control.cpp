#include "NMPC_control.h"
#include <algorithm>
#include <chrono>

namespace mpc_v2 {
namespace {
using Clock = std::chrono::steady_clock;

double milliseconds(
    Clock::time_point start,
    Clock::time_point end
)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}
} // namespace

Controller::Controller(
    Dynamics& dynamics,
    double timestep,
    int horizon,
    bool warm_start
)
    : dynamics(dynamics),
      N(checked_horizon(horizon)),
      Ts(timestep),
      nominal_state(N + 1),
      nominal_tau(N),
      prediction_tau(N),
      Mat_A(N),
      Mat_B(N),
      nominal_U(N * DOF),
      ref_U(N * DOF),
      state_error(N * NX),
      mpc_matrices(N),
      prediction_solver(N, warm_start)
{
    require(std::isfinite(Ts) && Ts > 0, "timestep must be positive");
}

void Controller::reset()
{
    first_control_cycle = true;
    prediction_solver.reset();
    last_timing = Timing{};
}

void Controller::set_torque_limits(
    const Joint& tau_lower,
    const Joint& tau_upper
)
{
    require(
        tau_lower.allFinite() && tau_upper.allFinite() && (tau_lower.array() < tau_upper.array()).all(),
        "Expected finite lower < upper torque limits"
    );
    this->tau_lower = tau_lower;
    this->tau_upper = tau_upper;
    reset();
}

Joint Controller::compute_control(
    const State& current_state,
    const Eigen::MatrixXd& state_ref,
    const Eigen::MatrixXd& ddq_ref
)
{
    const auto start = Clock::now();
    last_timing = Timing{};
    try {
        require(
            state_ref.rows() == N + 1 && state_ref.cols() == NX && ddq_ref.rows() == N && ddq_ref.cols() == DOF,
            "Reference shape mismatch"
        );
        require(
            current_state.allFinite() && state_ref.allFinite() && ddq_ref.allFinite(),
            "Nonfinite controller input"
        );

        // 1. 与旧版一样，用参考轨迹的 RNEA 生成参考力矩。
        for (int i = 0; i < N; ++i) {
            const State reference_state = state_ref.row(i).transpose();
            const Joint reference_ddq = ddq_ref.row(i).transpose();
            ref_U.segment<DOF>(i * DOF) = dynamics.inverse_dynamics(reference_state, reference_ddq);
            nominal_tau[i] =
                first_control_cycle ? Joint(ref_U.segment<DOF>(i * DOF)) : prediction_tau[std::min(i + 1, N - 1)];
            nominal_U.segment<DOF>(i * DOF) = nominal_tau[i];
        }
        const auto after_reference = Clock::now();
        last_timing.reference_ms = milliseconds(start, after_reference);

        // 2. 只积分一条名义轨迹，同时得到每个区间的 A、B。
        nominal_state[0] = current_state;
        for (int i = 0; i < N; ++i) {
            const auto linearization = dynamics.integrate_linearized(nominal_state[i], nominal_tau[i], Ts);
            nominal_state[i + 1] = linearization.next;
            Mat_A[i] = linearization.A;
            Mat_B[i] = linearization.B;
            state_error.segment<NX>(i * NX) = nominal_state[i + 1] - state_ref.row(i + 1).transpose();
        }
        const auto after_dynamics = Clock::now();
        last_timing.dynamics_ms = milliseconds(after_reference, after_dynamics);

        // 3. delta_x0=0，所以 delta_X = Gamma * delta_U。
        mpc_matrices.update(Mat_A, Mat_B);
        prediction_solver.prepare(mpc_matrices, state_error, nominal_U, ref_U, tau_lower, tau_upper);
        const auto after_matrices = Clock::now();
        last_timing.matrices_ms = milliseconds(after_dynamics, after_matrices);

        // 4. 求解器持续保存，H 变化时也能正确热启动。
        const Eigen::VectorXd& delta_U = prediction_solver.solve();
        last_timing.qp_ms = milliseconds(after_matrices, Clock::now());
        last_timing.qp_iterations = prediction_solver.iterations();
        for (int i = 0; i < N; ++i) {
            prediction_tau[i] = nominal_tau[i] + delta_U.segment<DOF>(i * DOF);
            if (!prediction_tau[i].allFinite() || (prediction_tau[i] - tau_lower).minCoeff() < -1e-5 ||
                (tau_upper - prediction_tau[i]).minCoeff() < -1e-5) {
                throw std::runtime_error("Invalid final torque sequence");
            }
        }
        first_control_cycle = false;
        last_timing.qp_success = true;
        last_timing.total_ms = milliseconds(start, Clock::now());
        return prediction_tau[0];
    } catch (...) {
        // 下次从参考重新初始化；当前必须把失败交给调用方处理。
        first_control_cycle = true;
        prediction_solver.reset();
        last_timing.qp_success = false;
        last_timing.total_ms = milliseconds(start, Clock::now());
        throw;
    }
}

} // namespace mpc_v2
