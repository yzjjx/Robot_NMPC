#ifndef MPC_V2_PREDICTION_H
#define MPC_V2_PREDICTION_H

#include "MPC_Matrices.h"
#include <qpOASES.hpp>
#include <array>

namespace mpc_v2 {

class Prediction {
public:
    explicit Prediction(
        int horizon,
        bool warm_start = true
    );
    Prediction(
        const Prediction&
    ) = delete;
    Prediction& operator=(
        const Prediction&
    ) = delete;
    void prepare(
        const MPCMatrices& mpc_matrices,
        const Eigen::VectorXd& state_error,
        const Eigen::VectorXd& nominal_U,
        const Eigen::VectorXd& ref_U,
        const Joint& tau_lower,
        const Joint& tau_upper
    );
    const Eigen::VectorXd& solve();
    void reset();

    int iterations() const
    {
        return qp_iterations;
    }

private:
    using QpMatrix = Eigen::Matrix<qpOASES::real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using QpVector = Eigen::Matrix<qpOASES::real_t, Eigen::Dynamic, 1>;
    int nV; // QP 决策变量数，等于 N*DOF
    // qpOASES 对 Hessian 使用浅拷贝。双份 H 保证写新 H 时旧 H 仍有效。
    std::array<QpMatrix, 2> H_buffer;
    QpVector g_qp;     // QP 一次项
    QpVector lb;       // 力矩增量下界
    QpVector ub;       // 力矩增量上界
    QpVector solution; // qpOASES 原始解

    Eigen::VectorXd weighted_state_error; // 状态误差乘以对应权重
    Eigen::VectorXd delta_U;              // 准备时复用为梯度缓存，求解后保存力矩增量
    qpOASES::SQProblem qp_solver;         // 声明在 H_buffer 后，析构时先释放求解器

    bool use_warm_start;             // 是否启用热启动
    bool solver_initialized = false; // 是否已有可用于热启动的解
    bool qp_prepared = false;        // 本周期 QP 数据是否已准备完成
    int active_hessian_index = -1;   // 求解器当前引用的 Hessian 缓存
    int pending_hessian_index = 0;   // 正在准备的新 Hessian 缓存
    int qp_iterations = 0;           // 求解前为迭代上限，求解后为实际工作集重算次数
};

} // namespace mpc_v2

#endif // MPC_V2_PREDICTION_H
