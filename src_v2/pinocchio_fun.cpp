#include "pinocchio_fun.h"

#include <algorithm>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>

namespace mpc_v2 {

Dynamics::Dynamics(const std::string& urdf, bool gravity_compensated, double integration_step)
    : data_(model_), max_step_(integration_step) {
    require(std::isfinite(max_step_) && max_step_ > 0, "integration_step must be positive");
    pinocchio::urdf::buildModel(urdf, model_);
    require(model_.nq == DOF && model_.nv == DOF, "Expected a 6-DOF fixed-base model");
    // q+v*h 只适用于这里的六个标量关节，不支持浮动基座/四元数关节。
    for (std::size_t i = 1; i < model_.joints.size(); ++i)
        require(model_.joints[i].nq() == 1 && model_.joints[i].nv() == 1,
                "Expected scalar joints");
    // 与旧版默认相同：输入为机器人底层补偿重力之后的附加力矩。
    // 只有确认接口接收总力矩时，才把 gravity_compensated 设为 false。
    if (gravity_compensated) model_.gravity.setZero();
    data_ = pinocchio::Data(model_);
}

int Dynamics::step_count(double duration) const {
    require(std::isfinite(duration) && duration >= 0, "duration must be finite and nonnegative");
    const double count = std::ceil(duration / max_step_);
    require(count <= 1000000, "Too many integration substeps");
    return std::max(1, static_cast<int>(count));
}

State Dynamics::advance(const State& x, const Joint& ddq, double h) {
    State next;
    next.head<DOF>() = x.head<DOF>() + h*x.tail<DOF>() + (0.5*h*h)*ddq;
    next.tail<DOF>() = x.tail<DOF>() + h*ddq;
    return next;
}

Joint Dynamics::acceleration(const State& x, const Joint& u) {
    require(x.allFinite() && u.allFinite(), "Nonfinite ABA input");
    Joint ddq = pinocchio::aba(model_, data_, x.head<DOF>(), x.tail<DOF>(), u);
    if (!ddq.allFinite()) throw std::runtime_error("Nonfinite ABA result");
    return ddq;
}

Joint Dynamics::inverse_dynamics(const State& x, const Joint& ddq) {
    require(x.allFinite() && ddq.allFinite(), "Nonfinite RNEA input");
    Joint u = pinocchio::rnea(model_, data_, x.head<DOF>(), x.tail<DOF>(), ddq);
    if (!u.allFinite()) throw std::runtime_error("Nonfinite RNEA result");
    return u;
}

State Dynamics::integrate(const State& x, const Joint& u, double duration) {
    require(x.allFinite() && u.allFinite(), "Nonfinite integration input");
    const int steps = step_count(duration);
    const double h = duration / steps;
    State current = x;
    if (duration == 0) return current;
    for (int j = 0; j < steps; ++j)
        current = advance(current, acceleration(current, u), h);
    if (!current.allFinite()) throw std::runtime_error("Nonfinite integrated state");
    return current;
}

Linearization Dynamics::integrate_linearized(const State& x, const Joint& u, double duration) {
    require(x.allFinite() && u.allFinite(), "Nonfinite linearization input");
    const int steps = step_count(duration);
    const double h = duration / steps;
    Linearization out{x, MatA::Identity(), MatB::Zero()};
    if (duration == 0) return out;
    const Eigen::Matrix<double, DOF, DOF> I = Eigen::Matrix<double, DOF, DOF>::Identity();

    for (int j = 0; j < steps; ++j) {
        // 先保存加速度，再计算同一个 (q,v,u) 点的解析导数。
        // 使用带 q/v/u 的完整接口，避免依赖不同 Pinocchio 版本的中间量复用约定。
        const Joint ddq = acceleration(out.next, u);
        pinocchio::computeABADerivatives(model_, data_, out.next.head<DOF>(),
                                       out.next.tail<DOF>(), u);
        const Eigen::Matrix<double, DOF, DOF> gq = data_.ddq_dq;
        const Eigen::Matrix<double, DOF, DOF> gv = data_.ddq_dv;
        // 某些版本只填 Minv 的上三角，必须恢复对称矩阵。
        const Eigen::Matrix<double, DOF, DOF> gu = data_.Minv.selfadjointView<Eigen::Upper>();

        // 对当前常加速度积分公式求导，而不是简单使用 I+h*Ac。
        MatA a;
        a.topLeftCorner<DOF, DOF>() = I + (0.5*h*h)*gq;
        a.topRightCorner<DOF, DOF>() = h*I + (0.5*h*h)*gv;
        a.bottomLeftCorner<DOF, DOF>() = h*gq;
        a.bottomRightCorner<DOF, DOF>() = I + h*gv;
        MatB b;
        b.topRows<DOF>() = (0.5*h*h)*gu;
        b.bottomRows<DOF>() = h*gu;

        // 链式法则：从小步的导数得到整个保持力矩区间的导数。
        out.A = (a*out.A).eval();
        out.B = (a*out.B + b).eval();
        out.next = advance(out.next, ddq, h);
        if (!out.next.allFinite() || !out.A.allFinite() || !out.B.allFinite())
            throw std::runtime_error("Nonfinite state or sensitivity");
    }
    return out;
}

} // namespace mpc_v2
