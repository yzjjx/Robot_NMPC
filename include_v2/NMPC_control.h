#ifndef MPC_V2_NMPC_CONTROL_H
#define MPC_V2_NMPC_CONTROL_H

#include "pinocchio_fun.h"
#include "Prediction.h"

namespace mpc_v2 {

// 简单 CPU 版本：测量 -> 名义积分与导数 -> 凝聚 -> QP -> 第一组力矩。
// 没有机器人通信、跨周期模型缓存、自动实物运行或安全控制器。
class Controller {
public:
    Controller(
        Dynamics& dynamics,
        double timestep = 0.001,
        int horizon = 40,
        bool warm_start = true
    );
    Joint compute_control(
        const State& current_state,
        const Eigen::MatrixXd& state_ref,
        const Eigen::MatrixXd& ddq_ref
    );
    void reset();
    void set_torque_limits(
        const Joint& tau_lower,
        const Joint& tau_upper
    );

    double timestep() const
    {
        return Ts;
    }

    int horizon() const
    {
        return N;
    }

    const Timing& timing() const
    {
        return last_timing;
    }

private:
    Dynamics& dynamics; // 动力学对象，与旧版 pinocchioFun 对应
    int N;              // 预测步数
    double Ts;          // 每个预测区间内保持力矩的时间

    bool first_control_cycle = true;        // 是否需要使用参考力矩初始化
    Joint tau_lower = Joint::Constant(-30); // 附加力矩下界
    Joint tau_upper = Joint::Constant(30);  // 附加力矩上界

    std::vector<State> nominal_state;  // 名义状态序列，N+1 个节点
    std::vector<Joint> nominal_tau;    // 名义力矩序列，N 个节点
    std::vector<Joint> prediction_tau; // QP 优化后的完整力矩序列
    std::vector<MatA> Mat_A;           // 各预测区间的离散状态矩阵
    std::vector<MatB> Mat_B;           // 各预测区间的离散输入矩阵

    Eigen::VectorXd nominal_U;   // 将 nominal_tau 叠加为 N*DOF 维向量
    Eigen::VectorXd ref_U;       // RNEA 计算的参考力矩向量
    Eigen::VectorXd state_error; // 名义预测状态与参考状态之差

    MPCMatrices mpc_matrices;     // 预测矩阵及权重缓存
    Prediction prediction_solver; // 持久化 QP 求解器
    Timing last_timing;           // 最近一次控制计算的耗时和状态
};

} // namespace mpc_v2

#endif // MPC_V2_NMPC_CONTROL_H
