#include "MPC_Matrices.h"

MPC_Matrices compute_mpc_matrices(
    const std::vector<Eigen::MatrixXd>& Mat_A,
    const std::vector<Eigen::MatrixXd>& Mat_B,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& F,
    int N
)
{
    int n = Mat_A[0].rows(); // 状态维度
    int p = Mat_B[0].cols(); // 控制输入维度

    // Delta X = phi*Delta_x0+Gamma*Delta_U
    // 这里n = 12， p = 6
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(N*n, n);
    Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(N*n, N*p);

    Eigen::MatrixXd Phi_i = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd Gamma_i = Eigen::MatrixXd::Zero(n, N*p);

    for(int i = 0; i < N; i++){
        //delta_x_i = A_i*delta_x_{i}+B_i*delta_u_i
        Phi_i = Mat_A[i] * Phi_i;
        Gamma_i = Mat_A[i] * Gamma_i;
        Gamma_i.block(0, i*p, n, p) += Mat_B[i];
        
        Phi.block(i*n, 0, n, n) = Phi_i;
        Gamma.block(i*n, 0, n, N*p) = Gamma_i;
    }

    // 状态权重使用Q，终端使用F，也就是说Q_bar的最后一行使用F
    Eigen::MatrixXd Q_bar = Eigen::MatrixXd::Zero(N*n, N*n);
    for(int i = 0; i < N-1; i++){
        Q_bar.block(i*n, i*n, n, n) = Q;
    }
    Q_bar.block((N-1)*n, (N-1)*n, n, n) = F;

    // 控制权重：R
    Eigen::MatrixXd R_bar = Eigen::MatrixXd::Zero(N*p, N*p);
    for(int i = 0; i < N; i++){
        R_bar.block(i*p, i*p, p, p) = R;
    }

    return {Phi, Gamma, Q_bar, R_bar};

}