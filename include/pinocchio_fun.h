#ifndef PINOCCHIO_FUN_H
#define PINOCCHIO_FUN_H

#include <Eigen/Dense>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <vector>

class pinocchioFun {
private:
    int DOF; // 自由度
    // 前向差分计算步长
    double q_step;
    double dq_step;
    double tau_step;

    pinocchio::Model model; // Pinocchio模型
    pinocchio::Data data; // Pinocchio数据
public:
    explicit pinocchioFun(const std::string& urdf_path);

    void set_forward_diff_step(
        double q_step,
        double dq_step,
        double tau_step
    );

    // RNEA输出不含重力的附加关节力矩；重力由机器人底层补偿。
    Eigen::VectorXd compute_rnea(
        const Eigen::VectorXd& q,
        const Eigen::VectorXd& dq,
        const Eigen::VectorXd& ddq
    );

    // ABA使用同一无重力模型，与常加速度离散化结合得到下一时刻的状态。
    Eigen::VectorXd compute_aba(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control,
        double Ts);

    // 力矩在 duration 内保持不变，用不超过 1 ms 的小步长积分。
    Eigen::VectorXd compute_held_state(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control,
        double duration);

    // 使用前向差分计算雅可比矩阵A和矩阵B
    void com_Mat_A_B(
        const std::vector<Eigen::VectorXd>& nom_state,
        const std::vector<Eigen::VectorXd>& mom_control,
        const std::vector<Eigen::VectorXd>& nom_state_next,
        double Ts,
        std::vector<Eigen::MatrixXd>& Mat_A,
        std::vector<Eigen::MatrixXd>& Mat_B
    );
};

#endif
