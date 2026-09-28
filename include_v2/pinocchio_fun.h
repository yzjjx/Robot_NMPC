#pragma once

#include "types.h"
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <string>

namespace mpc_v2 {

struct Linearization {
    State next;
    MatA A;
    MatB B;
};

// 一个 Dynamics 对象含有可变的 Pinocchio Data，不可跨线程同时调用。
class Dynamics {
public:
    explicit Dynamics(const std::string& urdf, bool gravity_compensated = true,
                      double integration_step = 0.001);

    Joint acceleration(const State& x, const Joint& u);
    Joint inverse_dynamics(const State& x, const Joint& ddq);
    State integrate(const State& x, const Joint& u, double duration);
    Linearization integrate_linearized(const State& x, const Joint& u, double duration);
    double integration_step() const { return max_step_; }

private:
    pinocchio::Model model_;
    pinocchio::Data data_;
    double max_step_;
    int step_count(double duration) const;
    static State advance(const State& x, const Joint& ddq, double h);
};

} // namespace mpc_v2
