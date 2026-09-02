#include "pinocchio_fun.h"

#include <cmath>

#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/parsers/urdf.hpp>

pinocchioFun::pinocchioFun(
    const std::string& urdf_path)
    :model(),data(model)
{
    DOF = 6; // 自由度
    q_step = 1e-6; // 前向差分计算步长
    dq_step = 1e-6;
    tau_step = 1e-4;
    
    pinocchio::urdf::buildModel(urdf_path, model);
    data = pinocchio::Data(model);
}

void pinocchioFun::set_forward_diff_step(
    double q_step_,
    double dq_step_,
    double tau_step_)
{
    q_step = q_step_;
    dq_step = dq_step_;
    tau_step = tau_step_;
}

Eigen::VectorXd pinocchioFun::compute_rnea(
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& dq,
    const Eigen::VectorXd& ddq)
{
    // 检查输入维度是否正确
    if (q.size() != DOF || dq.size() != DOF || ddq.size() != DOF) {
        throw std::invalid_argument("输入维度不正确.");
    }

    // 使用Pinocchio计算RNEA得到关节力矩
    Eigen::VectorXd tau = pinocchio::rnea(model, data, q, dq, ddq);
    
    return tau;
}

Eigen::VectorXd pinocchioFun::compute_aba(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    double Ts)
{
    // 检查输入维度是否正确
    if (state.size() != 2 * DOF || control.size() != DOF) {
        throw std::invalid_argument("输入维度不正确.");
    }

    // 将状态向量拆分为关节位置和速度
    Eigen::VectorXd q = state.head(DOF);
    Eigen::VectorXd dq = state.tail(DOF);

    // 使用Pinocchio计算ABA得到关节加速度
    Eigen::VectorXd ddq = pinocchio::aba(model, data, q, dq, control);

    // 使用半隐式欧拉离散化得到下一时刻的状态
    Eigen::VectorXd next_state(2 * DOF);
    next_state.head(DOF) = q + dq * Ts + 0.5 * ddq * Ts * Ts; // 更新关节位置
    next_state.tail(DOF) = dq + ddq * Ts; // 更新关节速度

    return next_state;
}

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

    for (int i = 0; i < N; i++) {
        const Eigen::VectorXd& x_bar = nom_state[i];
        const Eigen::VectorXd& u_bar = nom_control[i];
        const Eigen::VectorXd& x_bar_next = nom_state_next[i];

        for(int j = 0; j < x_n; j++) {
            Eigen::VectorXd x_perturbed = x_bar;
            x_perturbed(j) += q_step; // 对状态向量的每个元素进行扰动
            Eigen::VectorXd x_next_perturbed = compute_aba(x_perturbed, u_bar, Ts);
            Mat_A[i].col(j) = (x_next_perturbed - x_bar_next) / q_step; // 计算雅可比矩阵A的每一列
        }

        for(int j = 0; j < u_n; j++) {
            Eigen::VectorXd u_perturbed = u_bar;
            u_perturbed(j) += tau_step; // 对控制输入向量的每个元素进行扰动
            Eigen::VectorXd x_next_perturbed = compute_aba(x_bar, u_perturbed, Ts);
            Mat_B[i].col(j) = (x_next_perturbed - x_bar_next) / tau_step; // 计算雅可比矩阵B的每一列
        }
        
    }
}