#include "NMPC_control.h"
#include "pinocchio_fun.h"
#include "MPC_Matrices.h"
#include "Prediction.h"

#include <iostream>

ROKAE_NMPC::ROKAE_NMPC(pinocchioFun& dynamics)
    : dynamics(dynamics)
{
    // 控制器基本参数
    Ts = 0.01;  // 控制周期
    DOF = 6;    // 自由度
    x_n = 12;   // 状态维度,分别为关节位置和关节速度
    u_n = 6;    // 控制输入维度,关节力矩,这里控制输入指的是控制器输入到机器人里面的
    N = 20;     // 预测步长

    first_control_cycle = true;// 标记是否为第一次控制周期

    Eigen::VectorXd q_diag(x_n); // 状态权重矩阵对角线元素
    q_diag.head(DOF) << 100, 100, 100, 100, 100, 100; // 关节位置权重
    q_diag.tail(DOF) << 1, 1, 1, 1, 1, 1; // 关节速度权重
    Q = q_diag.asDiagonal(); // 状态权重矩阵

    F = 2.0*Q; // 终端状态权重矩阵
    R = 0.01*Eigen::MatrixXd::Identity(u_n, u_n); // 控制输入权重矩阵

    tau_lower = Eigen::VectorXd::Constant(u_n, -300); // 控制输入下界
    tau_upper = Eigen::VectorXd::Constant(u_n, 300); // 控制输入上界

    // 预测状态序列,21个12维状态向量
    nominal_state.resize(N+1, Eigen::VectorXd::Zero(x_n)); 
    // 创建20个控制向量，20个6维控制向量
    nominal_tau.resize(N, Eigen::VectorXd::Zero(u_n));

    // 保存20次名义ABA计算得到的下一时刻状态的结果
    nominal_state_next.resize(N, Eigen::VectorXd::Zero(x_n));

    Mat_A.resize(N, Eigen::MatrixXd::Zero(x_n, x_n)); // 线性化状态矩阵
    Mat_B.resize(N, Eigen::MatrixXd::Zero(x_n, u_n)); // 线性化控制矩阵

    delta_tau.resize(N, Eigen::VectorXd::Zero(u_n)); // 控制增量
    prediction_tau.resize(N, Eigen::VectorXd::Zero(u_n)); // 预测控制输入
    prediction_state.resize(N+1, Eigen::VectorXd::Zero(x_n)); // 预测状态序列
}

// 第一个控制周期使用目标状态的RNEA力矩初始化U_bar,也就是预测区域内部20个参考状态通过RNEA计算得到的参考力矩
void ROKAE_NMPC::initial_nom_ctrl(
    const std::vector<Eigen::VectorXd>& control_ref
)
{
    for(int i = 0; i < N; ++i) {
        nominal_tau[i] = control_ref[i];
    }
}

// 第二个及其以后的控制周期：将上一个周期内的最优控制序列向前移动一位
void ROKAE_NMPC::shift_nom_ctrl() {
    for(int i = 0; i < N-1; ++i) {
        nominal_tau[i] = prediction_tau[i+1];
    }
    // 如果最后没有新的控制量进入（没用新的期望状态输入），重复上一周期的最后一个控制量
    nominal_tau[N-1] = prediction_tau[N-1];
}

// 使用U_bar和完整非线性ABA计算得到的状态序列X_bar作为预测状态序列
void ROKAE_NMPC::generate_nom_traj(
    const Eigen::VectorXd& current_state
)
{
    // 将当前状态作为预测状态序列的第一个状态
    nominal_state[0] = current_state;

    // 使用ABA计算得到名义预测状态序列X_bar
    for(int i = 0; i < N; ++i) {
        nominal_state_next[i] = dynamics.compute_aba(
            nominal_state[i],
            nominal_tau[i],
            Ts
        );
        nominal_state[i+1] = nominal_state_next[i];
    }
}

// 复用生成X_bar的函数，使用ABA计算得到预测状态序列X_bar，进行前向差分
void ROKAE_NMPC::calculate_Mat_AB()
{
    dynamics.com_Mat_A_B(
        nominal_state,
        nominal_tau,
        nominal_state_next,
        Ts,
        Mat_A,
        Mat_B
    );
}

// 总控制函数
Eigen::VectorXd ROKAE_NMPC::compute_control(
    const Eigen::VectorXd& current_state,
    const std::vector<Eigen::VectorXd>& state_ref,
    const std::vector<Eigen::VectorXd>& ddq_ref
) {
    // 创建保存预测力矩的容器，20个6维控制向量
    std::vector<Eigen::VectorXd> control_ref(N, Eigen::VectorXd::Zero(u_n));
    // 使用RNEA计算参考力矩
    for(int i = 0;i < N;i++){
        control_ref[i] = dynamics.compute_rnea(
            state_ref[i].head(DOF), // 关节位置
            state_ref[i].tail(DOF), // 关节速度
            ddq_ref[i] // 关节加速度
        );
    }

    // 第一次使用RNEA计算得到的参考力矩初始化U_bar，后续周期使用上一个周期的最优控制序列向前移动一位
    if (first_control_cycle) {
        initial_nom_ctrl(control_ref);
        first_control_cycle = false;
    } else {
        shift_nom_ctrl();
    }

    // 20次ABA计算得到名义预测状态序列X_bar
    generate_nom_traj(current_state);

    // 前向差分计算A和B矩阵
    calculate_Mat_AB();

    // 构造Delta X = phi*delta_x0 +  gamma*delta_U
    MPC_Matrices matrices = compute_mpc_matrices(
        Mat_A,
        Mat_B,
        Q,
        R,
        F,
        N
    );

    Eigen::VectorXd nominal_X(N*x_n);
    Eigen::VectorXd nominal_U(N*u_n);
    Eigen::VectorXd ref_X(N*x_n);
    Eigen::VectorXd ref_U(N*u_n);

    for(int i = 0; i<N;i++){
        nominal_X.segment(i*x_n, x_n) = nominal_state[i+1];
        nominal_U.segment(i*u_n, u_n) = nominal_tau[i];
        ref_X.segment(i*x_n, x_n) = state_ref[i+1];
        ref_U.segment(i*u_n, u_n) = control_ref[i];
    }

    // 因为名义轨迹从最新状态开始，所以delta_x0 = 0
    Eigen::VectorXd delta_x0 = current_state - nominal_state[0];

    // 构造代价函数并且求解QP
    Eigen::VectorXd delta_U = Prediction(
        delta_x0,
        nominal_X,
        nominal_U,
        ref_X,
        ref_U,
        matrices,
        tau_lower,
        tau_upper,
        N,
        u_n
    );

    // U = U_bar + delta_U
    for(int i = 0;i<N;i++)
    {
        delta_tau[i] = delta_U.segment(i*u_n,u_n);
        prediction_tau[i] = nominal_tau[i] + delta_tau[i];
    }

    // X = X_bar +phi*delta_x0+gamma*delat U
    Eigen::VectorXd delat_X = matrices.Phi*delta_x0+matrices.Gamma*delta_U;

    prediction_state[0] = current_state;
    for(int i = 0;i<N;i++)
    {
        prediction_state[i+1] = nominal_state[i+1] + delat_X.segment(i*x_n,x_n);
    }

    // 滚动时域控制之执行第一步
    return prediction_tau[0];
}

