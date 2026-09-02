#ifndef NMPC_CONTROL_H
#define NMPC_CONTROL_H

#include <Eigen/Dense>
#include <vector>

class pinocchioFun;

class ROKAE_NMPC {
private:
    double Ts; // 控制周期
    int DOF;   // 自由度
    int x_n;   // 状态维度
    int u_n;   // 控制输入维度
    int N;     // 预测步长

    bool first_control_cycle; // 标记是否为第一次控制周期

    Eigen::MatrixXd Q; // 状态权重矩阵
    Eigen::MatrixXd F; // 终端状态权重矩阵
    Eigen::MatrixXd R; // 控制输入权重矩阵

    Eigen::VectorXd tau_lower; // 控制输入下界
    Eigen::VectorXd tau_upper; // 控制输入上界

    pinocchioFun& dynamics;

    std::vector<Eigen::VectorXd> nominal_state; // 预测状态序列
    std::vector<Eigen::VectorXd> nominal_tau; // 预测控制输入序
    std::vector<Eigen::VectorXd> nominal_state_next; // 保存名义ABA计算得到的下一时刻状态的结果 

    std::vector<Eigen::MatrixXd> Mat_A; // 线性化状态矩阵
    std::vector<Eigen::MatrixXd> Mat_B; // 线性化控制

    // QP求解得到delat_tau,然后更新预测控制输入和预测状态序列
    std::vector<Eigen::VectorXd> delta_tau; // 控制增量
    std::vector<Eigen::VectorXd> prediction_tau; // 预测控制输入
    std::vector<Eigen::VectorXd> prediction_state; // 预测状态序列

    void initial_nom_ctrl(const std::vector<Eigen::VectorXd>& control_ref); // 第一个控制周期使用目标状态的RNEA力矩初始化U_bar
    void shift_nom_ctrl(); // 第二个及其以后的控制周期：将上一个周期内的最优控制序列向前移动一位
    void linearize_dynamics(); // 线性化动力学模型，计算Mat_A和Mat_B
    void generate_nom_traj(const Eigen::VectorXd& current_state); // 生成名义轨迹

    void calculate_Mat_AB();

public:
    explicit ROKAE_NMPC(pinocchioFun& dynamics);

    //LTC-MPC控制接口，输入x_ref和ddq_ref，输出tau
    Eigen::VectorXd compute_control(
        const Eigen::VectorXd& current_state,
        const std::vector<Eigen::VectorXd>& state_ref,
        const std::vector<Eigen::VectorXd>& ddq_ref
    );

    // const std::vector<Eigen::VectorXd>& get

};

#endif