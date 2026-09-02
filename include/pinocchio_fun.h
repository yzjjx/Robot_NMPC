#ifndef PINOCCHIO_FUN_H
#define PINOCCHIO_FUN_H

#include <eigen3/Eigen/Dense>
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

    // 使用Pinocchio计算RNEA得到关节力矩，写成函数主要用于判断维度是否正确
    Eigen::VectorXd compute_rnea(
        const Eigen::VectorXd& q,
        const Eigen::VectorXd& dq,
        const Eigen::VectorXd& ddq
    );

    // ABA计算与半隐式欧拉离散化结合，得到下一时刻的状态
    Eigen::VectorXd compute_aba(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control,
        double Ts);

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