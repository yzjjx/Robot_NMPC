#include "NMPC_control.h"
#include "pinocchio_fun.h"

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    const double pi = 3.14159265358979323846;
    const double rad_to_deg = 180.0 / pi;
    const double step_angle_deg = 30.0;    // 关节1阶跃角度

    // 第一个参数可以指定URDF，第二个参数可以指定输出文件
    std::string urdf_filename = (argc >= 2)
        ? argv[1]
        : "Robot_NMPC/urdf/ROKAE_SR4.urdf";
    std::string output_filename = (argc >= 3)
        ? argv[2]
        : "step_response.txt";

    try {
        // 1. 创建SR4动力学模型和LTV-MPC控制器
        pinocchioFun robot_dynamics(urdf_filename);
        robot_dynamics.set_forward_diff_step(1e-6, 1e-6, 1e-4);

        ROKAE_NMPC controller(robot_dynamics);
        // 从控制器读取参数，避免仿真周期与 MPC 周期不一致。
        const double Ts = controller.timestep();
        const int prediction_horizon = controller.horizon();
        const int simulation_steps = static_cast<int>(std::lround(3.0 / Ts));
        const int print_interval = static_cast<int>(std::lround(0.2 / Ts));

        // 2. 初始状态：六个关节的位置和速度全部为零
        Eigen::VectorXd current_state = Eigen::VectorXd::Zero(12);

        // 3. 给关节1施加从0度到30度的阶跃参考，其他关节参考保持0度
        Eigen::VectorXd q_ref = Eigen::VectorXd::Zero(6);
        q_ref(0) = step_angle_deg / rad_to_deg;

        Eigen::VectorXd ref_state = Eigen::VectorXd::Zero(12);
        ref_state.head(6) = q_ref;

        std::vector<Eigen::VectorXd> state_reference(
            prediction_horizon + 1,
            ref_state);
        std::vector<Eigen::VectorXd> acceleration_reference(
            prediction_horizon,
            Eigen::VectorXd::Zero(6));

        // 4. 创建阶跃响应输出文件
        std::ofstream output_file(output_filename);
        if(!output_file.is_open()) {
            throw std::runtime_error("无法创建阶跃响应输出文件.");
        }

        output_file << std::fixed << std::setprecision(10);
        output_file
            << "time_s q1_ref_deg "
            << "q1_deg q2_deg q3_deg q4_deg q5_deg q6_deg "
            << "tau1_Nm tau2_Nm tau3_Nm tau4_Nm tau5_Nm tau6_Nm"
            << std::endl;

        // 记录t=0时刻：关节位置为零，此时还没有施加控制力矩
        output_file << 0.0 << " " << step_angle_deg;
        for(int joint = 0; joint < 6; joint++) {
            output_file << " " << current_state(joint) * rad_to_deg;
        }
        for(int joint = 0; joint < 6; joint++) {
            output_file << " " << 0.0;
        }
        output_file << std::endl;

        std::cout << "Use robot model: " << urdf_filename << std::endl;
        std::cout << "Joint 1 step reference: "
                  << step_angle_deg << " deg" << std::endl;
        std::cout << "Control frequency: " << 1.0 / Ts << " Hz" << std::endl;

        // 5. 闭环运行3秒；1000 Hz时共有3000个控制周期
        for(int step = 0; step < simulation_steps; step++) {
            // MPC输入当前状态，输出当前周期应该执行的六维关节力矩
            Eigen::VectorXd control = controller.compute_control(
                current_state,
                state_reference,
                acceleration_reference);

            // 使用与控制器相同的周期模拟机器人向前运动一步
            current_state = robot_dynamics.compute_aba(
                current_state,
                control,
                Ts);

            double current_time = (step + 1) * Ts;

            // 每一行保存：时间、关节1参考角度、六关节实际角度、六关节力矩
            output_file << current_time << " " << step_angle_deg;
            for(int joint = 0; joint < 6; joint++) {
                output_file << " " << current_state(joint) * rad_to_deg;
            }
            for(int joint = 0; joint < 6; joint++) {
                output_file << " " << control(joint);
            }
            output_file << std::endl;

            // 每0.2秒在终端显示一次关节1的响应
            if((step + 1) % print_interval == 0) {
                std::cout << "time = " << current_time
                          << " s, q1 = "
                          << current_state(0) * rad_to_deg
                          << " deg, tau1 = "
                          << control(0)
                          << " Nm"
                          << std::endl;
            }
        }

        output_file.close();
        std::cout << "Step response saved to: "
                  << output_filename << std::endl;
    }
    catch(const std::exception& error) {
        std::cerr << "LTV-MPC运行失败：" << error.what() << std::endl;
        return 1;
    }

    return 0;
}
