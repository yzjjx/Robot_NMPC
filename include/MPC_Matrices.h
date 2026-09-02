#ifndef MPC_MATRICES_H
#define MPC_MATRICES_H

#include <Eigen/Dense>
#include <vector>

struct MPC_Matrices
{
    Eigen::MatrixXd Phi; // 状态转移矩阵
    Eigen::MatrixXd Gamma; // 控制输入矩阵
    Eigen::MatrixXd Q_bar; // 状态权重矩阵
    Eigen::MatrixXd R_bar; // 控制输入权重矩阵
};

MPC_Matrices compute_mpc_matrices(
    const std::vector<Eigen::MatrixXd>& Mat_A,
    const std::vector<Eigen::MatrixXd>& Mat_B,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& F,
    int N
);

# endif