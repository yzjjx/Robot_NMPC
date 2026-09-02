#include "Prediction.h"

#include <qpOASES.hpp>
#include <iostream>
#include <vector>

// Eigen默认为列优先，而qpOASES要求输入为行优先，因此需要将Eigen矩阵转换为行优先存储
template<typename Derived>//表明函数为模板函数，可以接受多种Eigen类型的输入
std::vector<qpOASES::real_t> Eigen2QpArray(
    const Eigen::MatrixBase<Derived>& eigen_matrix
)
{
    std::vector<qpOASES::real_t> data;
    data.reserve(eigen_matrix.size());

    for (int i = 0; i < eigen_matrix.rows(); i++) {
        for (int j = 0; j < eigen_matrix.cols(); j++) {
            data.push_back(static_cast<qpOASES::real_t>(eigen_matrix(i, j)));
        }
    }
    return data;
}

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
) 
{
    int nV = N*p;

    // 如果名义起点不等于实际测量状态，phi*delta_x0会修正所有状态预测
    Eigen::VectorXd state_error = nominal_X + mpc_matrices.Phi * delta_x0 - ref_X;
    Eigen::VectorXd control_err = nominal_U - ref_U;

    // qpOASES格式
    Eigen::MatrixXd H_qp = 2.0 * (mpc_matrices.Gamma.transpose()*mpc_matrices.Q_bar*mpc_matrices.Gamma+mpc_matrices.R_bar);
    H_qp += 1e-6*Eigen::MatrixXd::Identity(nV,nV);

    Eigen::MatrixXd g_qp = 2.0 * (mpc_matrices.Gamma.transpose()*mpc_matrices.Q_bar*state_error+mpc_matrices.R_bar*control_err);

    // 实际力矩计算
    Eigen::VectorXd lb = Eigen::VectorXd::Zero(nV);
    Eigen::VectorXd ub = Eigen::VectorXd::Zero(nV);
    for(int i = 0;i<N;i++)
    {
        lb.segment(i*p,p) = tau_lower-nominal_U.segment(i*p,p);
        ub.segment(i*p,p) = tau_upper-nominal_U.segment(i*p,p);
    }

    std::vector<qpOASES::real_t> H_arr = Eigen2QpArray(H_qp);
    std::vector<qpOASES::real_t> g_arr = Eigen2QpArray(g_qp);
    std::vector<qpOASES::real_t> lb_arr = Eigen2QpArray(lb);
    std::vector<qpOASES::real_t> ub_arr = Eigen2QpArray(ub);

    qpOASES::QProblemB problem(nV);
    qpOASES::Options options;
    options.printLevel = qpOASES::PL_NONE;
    problem.setOptions(options);

    int nWSR = 500;
    qpOASES::returnValue status = problem.init(
        H_arr.data(),
        g_arr.data(),
        lb_arr.data(),
        ub_arr.data(),
        nWSR
    );

    Eigen::VectorXd delta_U = Eigen::VectorXd::Zero(nV);

    if(status == qpOASES::SUCCESSFUL_RETURN)
    {
        std::vector<qpOASES::real_t> solution(nV);
        problem.getPrimalSolution(solution.data());
        delta_U = Eigen::Map<Eigen::VectorXd>(solution.data(),nV);
    }
    else{
        std::cerr<<"求解失败，错误码："<<static_cast<int>(status)<<std::endl;
    }

    return delta_U;
}