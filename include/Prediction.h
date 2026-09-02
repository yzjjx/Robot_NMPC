#ifndef PREDICTION_H
#define PREDICTION_H

#include "MPC_Matrices.h"
#include <Eigen/Dense>

// 求解QP问题
Eigen::VectorXd Prediction(
    const Eigen::VectorXd& delta_x0,
    const Eigen::VectorXd& nominal_X,
    const Eigen::VectorXd& nominal_U,
    const Eigen::VectorXd& ref_X,
    const Eigen::VectorXd& ref_U,
    const MPC_Matrices& mpc_matrices,
    const Eigen::VectorXd& tau_lower,
    const Eigen::VectorXd& tau_upper,
    int N,
    int p
);

#endif