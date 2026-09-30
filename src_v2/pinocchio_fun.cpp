#include "pinocchio_fun.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>

namespace mpc_v2 {

Dynamics::Dynamics(
    const std::string& urdf_path,
    bool /* gravity_compensated：仅保留参数以兼容现有调用，不再使用 */,
    double integration_step
)
    : data(model),
      max_integration_step(integration_step)
{
    // 无意义的判断
    if (!std::isfinite(max_integration_step) || max_integration_step <= 0) {
        throw std::invalid_argument("integration_step must be positive");
    }

    pinocchio::urdf::buildModel(urdf_path, model);

    // 无意义的判断
    if (model.nq != DOF || model.nv != DOF) {
        throw std::invalid_argument("Expected a 6-DOF fixed-base model");
    }

    // q+v*h 只适用于这里的六个标量关节，不支持浮动基座/四元数关节。
    for (std::size_t i = 1; i < model.joints.size(); ++i) {
        if (model.joints[i].nq() != 1 || model.joints[i].nv() != 1) {
            throw std::invalid_argument("Expected scalar joints");
        }
    }

    // 机器人底层始终补偿重力，控制器只计算附加力矩。
    // RNEA、ABA 和解析线性化统一关闭模型重力项，避免重复补偿。
    model.gravity.setZero();
    data = pinocchio::Data(model);
}

// 这一步代替函数Eigen::VectorXd pinocchioFun::compute_held_state，得到steps
int Dynamics::step_count(
    double duration
) const
{
    if (!std::isfinite(duration) || duration < 0) {
        throw std::invalid_argument("duration must be finite and nonnegative");
    }

    const double count = std::ceil(duration / max_integration_step);

    if (!std::isfinite(count) || count > 1000000) {
        throw std::invalid_argument("Too many integration substeps");
    }

    return std::max(1, static_cast<int>(count));
}

// 相当于函数compute_aba的 Eigen::VectorXd next_state(2 * DOF);这一部分，用来计算下一时刻的状态
State Dynamics::advance(
    const State& state,
    const Joint& ddq,
    double integration_dt
)
{
    State next_state;

    next_state.head<DOF>() =
        state.head<DOF>() + integration_dt * state.tail<DOF>() + (0.5 * integration_dt * integration_dt) * ddq;
    next_state.tail<DOF>() = state.tail<DOF>() + integration_dt * ddq;

    return next_state;
}

// ABA算法计算加速度
Joint Dynamics::acceleration(
    const State& state,
    const Joint& control
)
{
    if (!state.allFinite() || !control.allFinite()) {
        throw std::invalid_argument("Nonfinite ABA input");
    }

    Joint ddq = pinocchio::aba(model, data, state.head<DOF>(), state.tail<DOF>(), control);

    if (!ddq.allFinite()) {
        throw std::runtime_error("Nonfinite ABA result");
    }

    return ddq;
}

// RNEA计算
Joint Dynamics::inverse_dynamics(
    const State& state,
    const Joint& ddq
)
{
    if (!state.allFinite() || !ddq.allFinite()) {
        throw std::invalid_argument("Nonfinite RNEA input");
    }

    Joint control = pinocchio::rnea(model, data, state.head<DOF>(), state.tail<DOF>(), ddq);

    if (!control.allFinite()) {
        throw std::runtime_error("Nonfinite RNEA result");
    }

    return control;
}

// 相当于之前的函数compute_held_state
State Dynamics::integrate(
    const State& state,
    const Joint& control,
    double duration
)
{
    // 判断 无意义
    if (!state.allFinite() || !control.allFinite()) {
        throw std::invalid_argument("Nonfinite integration input");
    }

    const int steps = step_count(duration);
    const double integration_dt = duration / steps;
    State predicted_state = state;

    if (duration == 0) {
        return predicted_state;
    }

    // ABA计算
    for (int j = 0; j < steps; ++j) {
        predicted_state = advance(predicted_state, acceleration(predicted_state, control), integration_dt);
    }

    // 判断 无意义
    if (!predicted_state.allFinite()) {
        throw std::runtime_error("Nonfinite integrated state");
    }

    return predicted_state;
}

Linearization Dynamics::integrate_linearized(
    const State& state,
    const Joint& control,
    double duration
)
{
    if (!state.allFinite() || !control.allFinite()) {
        throw std::invalid_argument("Nonfinite linearization input");
    }
    const int steps = step_count(duration);
    const double integration_dt = duration / steps;
    Linearization linearization{state, MatA::Identity(), MatB::Zero()};
    if (duration == 0) {
        return linearization;
    }
    const Eigen::Matrix<double, DOF, DOF> identity = Eigen::Matrix<double, DOF, DOF>::Identity();

    for (int j = 0; j < steps; ++j) {
        // 先保存加速度，再计算同一个 (q,v,u) 点的解析导数。
        // 使用带 q/v/u 的完整接口，避免依赖不同 Pinocchio 版本的中间量复用约定。
        const Joint ddq = acceleration(linearization.next, control);
        pinocchio::computeABADerivatives(
            model,
            data,
            linearization.next.head<DOF>(),
            linearization.next.tail<DOF>(),
            control
        );
        const Eigen::Matrix<double, DOF, DOF> ddq_dq = data.ddq_dq;
        const Eigen::Matrix<double, DOF, DOF> ddq_dv = data.ddq_dv;
        // 某些版本只填 Minv 的上三角，必须恢复对称矩阵。
        const Eigen::Matrix<double, DOF, DOF> ddq_dtau = data.Minv.selfadjointView<Eigen::Upper>();

        // 对当前常加速度积分公式求导，而不是简单使用 I+h*Ac。
        MatA Mat_A_step;
        Mat_A_step.topLeftCorner<DOF, DOF>() = identity + (0.5 * integration_dt * integration_dt) * ddq_dq;
        Mat_A_step.topRightCorner<DOF, DOF>() =
            integration_dt * identity + (0.5 * integration_dt * integration_dt) * ddq_dv;
        Mat_A_step.bottomLeftCorner<DOF, DOF>() = integration_dt * ddq_dq;
        Mat_A_step.bottomRightCorner<DOF, DOF>() = identity + integration_dt * ddq_dv;
        MatB Mat_B_step;
        Mat_B_step.topRows<DOF>() = (0.5 * integration_dt * integration_dt) * ddq_dtau;
        Mat_B_step.bottomRows<DOF>() = integration_dt * ddq_dtau;

        // 链式法则：从小步的导数得到整个保持力矩区间的导数。
        linearization.A = (Mat_A_step * linearization.A).eval();
        linearization.B = (Mat_A_step * linearization.B + Mat_B_step).eval();
        linearization.next = advance(linearization.next, ddq, integration_dt);
        if (!linearization.next.allFinite() || !linearization.A.allFinite() || !linearization.B.allFinite()) {
            throw std::runtime_error("Nonfinite state or sensitivity");
        }
    }
    return linearization;
}

} // namespace mpc_v2
