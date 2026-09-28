#pragma once

#include "pinocchio_fun.h"
#include "Prediction.h"

namespace mpc_v2 {

// 简单 CPU 版本：测量 -> 名义积分与导数 -> 凝聚 -> QP -> 第一组力矩。
// 没有机器人通信、跨周期模型缓存、自动实物运行或安全控制器。
class Controller {
public:
    Controller(Dynamics& dynamics, double timestep = 0.001, int horizon = 40,
               bool warm_start = true);
    Joint compute_control(const State& state, const Eigen::MatrixXd& state_ref,
                          const Eigen::MatrixXd& ddq_ref);
    void reset();
    void set_torque_limits(const Joint& lower, const Joint& upper);
    double timestep() const { return Ts_; }
    int horizon() const { return N_; }
    const Timing& timing() const { return timing_; }

private:
    Dynamics& dynamics_;
    int N_;
    double Ts_;
    bool first_ = true;
    Joint lower_ = Joint::Constant(-30), upper_ = Joint::Constant(30);
    std::vector<State> x_;
    std::vector<Joint> u_, optimal_u_;
    std::vector<MatA> A_;
    std::vector<MatB> B_;
    Eigen::VectorXd nominal_u_, reference_u_, state_error_;
    MPCMatrices matrices_;
    Prediction qp_;
    Timing timing_;
};

} // namespace mpc_v2
