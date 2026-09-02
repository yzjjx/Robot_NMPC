#include "NMPC_control.h"
#include "pinocchio_fun.h"

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    // 不传参数时，默认使用当前工程中的SR4 URDF
    std::string urdf_filename = (argc >= 2)
        ? argv[1]
        : "urdf/ROKAE_SR4.urdf";

    try {
        // 1. 创建SR4动力学模型和LTV-MPC控制器
        pinocchioFun robot_dynamics(urdf_filename);
        robot_dynamics.set_forward_diff_step(1e-6, 1e-6, 1e-4);

        ROKAE_NMPC controller(robot_dynamics);

        // 2. 输入机器人当前状态：[q1...q6,dq1...dq6]
        Eigen::VectorXd current_state = Eigen::VectorXd::Zero(12);
        current_state(1) = 30.0 * 3.14159265358979323846 / 180.0;
        current_state(2) = 30.0 * 3.14159265358979323846 / 180.0;

        // 3. 参考轨迹：本例21个参考状态都取零，因此表现为定点控制
        Eigen::VectorXd q_ref = Eigen::VectorXd::Zero(6);
        Eigen::VectorXd ref_state = Eigen::VectorXd::Zero(12);
        ref_state.head(6) = q_ref;

        std::vector<Eigen::VectorXd> state_reference(
            21,
            ref_state);
        std::vector<Eigen::VectorXd> acceleration_reference(
            20,
            Eigen::VectorXd::Zero(6));

        std::cout << "Use robot model: " << urdf_filename << std::endl;
        std::cout << "Start ABA finite-difference LTV-MPC." << std::endl;

        // 4. 闭环运行300个控制周期
        for(int step = 0; step < 300; step++) {
            // 输入current_state，输出当前周期应该执行的6维关节力矩
            Eigen::VectorXd control = controller.compute_control(
                current_state,
                state_reference,
                acceleration_reference);

            // 这里用相同的ABA模型模拟真实机器人前进一步
            current_state = robot_dynamics.compute_aba(
                current_state,
                control,
                0.01);

            if(step % 20 == 0) {
                std::cout << "step " << step
                          << ", q = "
                          << current_state.head(6).transpose()
                          << std::endl;
            }
        }
    }
    catch(const std::exception& error) {
        std::cerr << "LTV-MPC运行失败：" << error.what() << std::endl;
        return 1;
    }

    return 0;
}

