#pragma once
#include "../include_v2/pinocchio_fun.h"
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace mpc_v3 {
using mpc_v2::State;
using mpc_v2::Joint;
using mpc_v2::MatA;
using mpc_v2::MatB;
using mpc_v2::Linearization;
using mpc_v2::require;
constexpr int NX = 12, NU = 6;
using Clock = std::chrono::steady_clock;
inline double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}

struct Config {
    int horizon = 25;
    double model_dt = 0.01;
    int feedback_ratio = 3; // 1:普通频率；3:每个模型周期有三个独立反馈QP
    double integration_step = 0.001;
    bool gravity_compensated = true;
    double torque_limit = 10.0; // CPU/FPGA公平对比统一默认±10；不是实物安全认证
    double torque_trust = 5.0;  // 每节点相对名义力矩的最大改变量
    double position_trust = 0.15;
    double velocity_trust = 0.5;
    double max_position_defect = 0.1;
    double max_velocity_defect = 1.0;
    double max_measurement_age = 0.001;
    bool enforce_feedback_budget = true;
    int qp_iterations = 500;
    double feedback_dt() const { return model_dt/feedback_ratio; }
    void validate() const {
        mpc_v2::checked_horizon(horizon);
        require(feedback_ratio >= 1 && feedback_ratio <= 8, "feedback_ratio must be 1..8");
        for (double v : {model_dt,integration_step,torque_limit,torque_trust,position_trust,
                         velocity_trust,max_position_defect,max_velocity_defect,max_measurement_age})
            require(std::isfinite(v) && v > 0, "Configuration values must be finite and positive");
        require(qp_iterations > 0, "qp_iterations must be positive");
    }
};

struct BackendStats {
    std::uint64_t batches = 0, samples = 0;
    double transfer_and_wait_ms = 0;
};

struct PreparationInfo {
    std::int64_t block = -1;
    double model_ms = 0, matrices_and_qp_ms = 0, total_ms = 0;
    double max_position_defect = 0, max_velocity_defect = 0;
    BackendStats backend;
};

struct FeedbackResult {
    std::int64_t tick = -1, block = -1;
    int phase = 0, qp_iterations = 0;
    Joint torque = Joint::Zero();
    Eigen::MatrixXd states;   // 行0是当前测量；行1..N对应本块固定的未来节点
    Eigen::MatrixXd controls;
    Eigen::VectorXd state_times;
    double solve_ms = 0, total_ms = 0;
    bool deadline_missed = false;
};
} // namespace mpc_v3
