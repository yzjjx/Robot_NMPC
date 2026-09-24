#include "pinocchio_fun.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/parsers/urdf.hpp>

// 需要在h文件新建一个pinocchioFun的类class
// 这个类有三个核心功能，分别为rnea计算、aba计算与矩阵计算

// 首先为构造函数，构造函数是创建一个对象时，自动执行的一段初始化程序
pinocchioFun::pinocchioFun(
    const std::string& urdf_path)
    :model(),data(model)
{
    DOF = 6; // 自由度
    q_step = 1e-6; // 前向差分计算步长
    dq_step = 1e-6;
    tau_step = 1e-4;
    
    pinocchio::urdf::buildModel(urdf_path, model);

    // 机器人底层始终补偿重力；RNEA、ABA及其线性化统一使用附加力矩。
    // tau = M(q) * ddq + C(q, dq) * dq，不再包含 g(q)。
    model.gravity.setZero();
    data = pinocchio::Data(model);
}

// 构造函数的部分扩展，构造函数为初始化参数，这部分的函数可以修改构造函数的大小
void pinocchioFun::set_forward_diff_step(
    double q_step_,
    double dq_step_,
    double tau_step_)
{
    q_step = q_step_;
    dq_step = dq_step_;
    tau_step = tau_step_;
}

// RNEA计算
Eigen::VectorXd pinocchioFun::compute_rnea(
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& dq,
    const Eigen::VectorXd& ddq)
{
    // 检查输入维度是否正确
    if (q.size() != DOF || dq.size() != DOF || ddq.size() != DOF) {
        throw std::invalid_argument("输入维度不正确.");
    }
    if (!q.allFinite() || !dq.allFinite() || !ddq.allFinite()) {
        throw std::invalid_argument("RNEA inputs must be finite.");
    }

    // 使用Pinocchio计算RNEA得到关节力矩
    Eigen::VectorXd tau = pinocchio::rnea(model, data, q, dq, ddq);
    
    return tau;
}


// ABA计算，ABA是根据当前q、dq和tau输出ddq
Eigen::VectorXd pinocchioFun::compute_aba(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    double Ts)
{
    // 检查输入维度是否正确
    if (state.size() != 2 * DOF || control.size() != DOF) {
        throw std::invalid_argument("输入维度不正确.");
    }
    if (!state.allFinite() || !control.allFinite() || !std::isfinite(Ts) || Ts < 0) {
        throw std::invalid_argument("ABA inputs must be finite and timestep nonnegative.");
    }

    // 将状态向量拆分为关节位置和速度
    Eigen::VectorXd q = state.head(DOF);
    Eigen::VectorXd dq = state.tail(DOF);

    // 使用Pinocchio计算ABA得到关节加速度，control就是tau
    Eigen::VectorXd ddq = pinocchio::aba(model, data, q, dq, control);

    // 按当前加速度作常加速度离散化，得到下一时刻的状态
    Eigen::VectorXd next_state(2 * DOF);
    next_state.head(DOF) = q + dq * Ts + 0.5 * ddq * Ts * Ts; // 更新关节位置
    next_state.tail(DOF) = dq + ddq * Ts; // 更新关节速度

    return next_state;
}


// 一个预测区间内保持力矩不变，以不超过 1 ms 的步长积分
// 给定当前状态和一个固定力矩，预测duration秒后机器人到哪里
// 在一个预测区间内，控制输入保持不变，即零阶保持器
Eigen::VectorXd pinocchioFun::compute_held_state(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    double duration)
{
    if(!std::isfinite(duration) || duration < 0) {
        throw std::invalid_argument("Prediction duration not ok");
    }
    // ceil是向上取整
    const int steps = std::max(1, static_cast<int>(std::ceil(duration / 0.001)));
    // 预测加速度ddq
    Eigen::VectorXd predicted = state;
    for(int i = 0; i < steps; ++i) {
        predicted = compute_aba(predicted, control, duration / steps);
    }
    return predicted;
}

// 矩阵计算
void pinocchioFun::com_Mat_A_B(
    const std::vector<Eigen::VectorXd>& nom_state,
    const std::vector<Eigen::VectorXd>& nom_control,
    const std::vector<Eigen::VectorXd>& nom_state_next,
    double Ts,
    std::vector<Eigen::MatrixXd>& Mat_A,
    std::vector<Eigen::MatrixXd>& Mat_B
)
{
    int N = static_cast<int>(nom_control.size()); // 预测步长
    int x_n = nom_state[0].size(); // 状态维度12
    int u_n = nom_control[0].size(); // 控制输入维度6

    // 取出当前这个预测步长的状态矩阵（12）和控制输入（6）
    for (int i = 0; i < N; i++) {
        const Eigen::VectorXd& x_bar = nom_state[i];
        const Eigen::VectorXd& u_bar = nom_control[i];
        const Eigen::VectorXd& x_bar_next = nom_state_next[i];

        for(int j = 0; j < x_n; j++) {
            const double step = (j < DOF) ? q_step : dq_step;
            Eigen::VectorXd x_perturbed = x_bar;
            x_perturbed(j) += step; // 位置、速度分别使用各自的差分步长
            Eigen::VectorXd x_next_perturbed = compute_held_state(x_perturbed, u_bar, Ts);
            Mat_A[i].col(j) = (x_next_perturbed - x_bar_next) / step; // 计算雅可比矩阵A的每一列
        }

        for(int j = 0; j < u_n; j++) {
            Eigen::VectorXd u_perturbed = u_bar;
            u_perturbed(j) += tau_step; // 对控制输入向量的每个元素进行扰动
            Eigen::VectorXd x_next_perturbed = compute_held_state(x_bar, u_perturbed, Ts);
            Mat_B[i].col(j) = (x_next_perturbed - x_bar_next) / tau_step; // 计算雅可比矩阵B的每一列
        }
        
    }
}
