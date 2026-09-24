// 读取圆轨迹：测量关节状态 -> MPC计算力矩 -> 发送力矩。
#include "NMPC_control.h"
#include "pinocchio_fun.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "rokae/robot.h"
#include <chrono>

using namespace rokae;

struct LogRow
{
    int index = 0;
    std::array<double, 6> position{};
    std::array<double, 6> velocity{};
    std::array<double, 6> torque{};
};

void writeLog(const std::string& filename,
              const std::vector<LogRow>& logs,
              int recorded_count)
{
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("无法创建实际关节状态文件");
    }

    file << "index,vel,pos,tau\n";
    file << std::setprecision(10);
    for (int row_index = 0; row_index < recorded_count; ++row_index) {
        const LogRow& row = logs[row_index];
        file << row.index << ",";

        const auto write_array = [&](const std::array<double, 6>& values) {
            file << "[";
            for (int joint = 0; joint < 6; ++joint) {
                if (joint > 0) file << ",";
                file << values[joint];
            }
            file << "]";
        };

        write_array(row.velocity);
        write_array(row.position);
        write_array(row.torque);
        file << "\n";
    }
}

int main()
{
    const double control_dt = 0.05; // MPC控制周期：50ms，即20Hz
    const double sample_dt = 0.001; // 轨迹采样周期和SDK发送周期：1ms

    // 在输入的期望轨迹中隔着reference_stride行取一个参考点
    const int reference_stride = static_cast<int>(std::lround(control_dt / sample_dt));
    int solve_count = 0;
    int overrun_count = 0;
    double total_solve_ms = 0.0;
    double max_solve_ms = 0.0;
    int exit_code = 0;
    const std::string input_q = "../data_in/circle_R200_joint_trajectory_SR4_V50.txt";
    const std::string output_txt = "../data_out/circle_R200_SR4.txt";

    try
    {
        // 文件每行：时间(s)、6个关节位置(rad)、6个关节速度(rad/s)。
        std::ifstream file(input_q);
        if (!file) {
            throw std::runtime_error("无法打开轨迹文件：" + input_q);
        }
        std::string header;
        std::getline(file, header);
        std::vector<Eigen::VectorXd> trajectory;
        double time;
        while (file >> time) {
            Eigen::VectorXd state(12);
            for (int j = 0; j < 12; ++j) {
                file >> state(j);
            }
            if (!file) {
                throw std::runtime_error("轨迹数据不完整");
            }
            trajectory.push_back(state);
        }
        if (trajectory.size() < 2) {
            throw std::runtime_error("轨迹至少需要两个点");
        }

        const int last = static_cast<int>(trajectory.size()) - 1;
        std::vector<Eigen::VectorXd> acceleration(trajectory.size());
        for (int i = 0; i < last; ++i) {
            acceleration[i] = (trajectory[i + 1].tail(6) - trajectory[i].tail(6)) / sample_dt;
        }
        acceleration[last] = Eigen::VectorXd::Zero(6);

        //----------------------------------------------------------------
        // 输入动力学模型和控制器
        pinocchioFun dynamics("../urdf/ROKAE_SR4.urdf");
        ROKAE_NMPC controller(dynamics, control_dt, 14);
        //----------------------------------------------------------------

        const int N = controller.horizon();
        std::vector<Eigen::VectorXd> state_ref(N + 1, Eigen::VectorXd::Zero(12));
        std::vector<Eigen::VectorXd> ddq_ref(N, Eigen::VectorXd::Zero(6));
        Eigen::VectorXd current_state(12);

        // 机器人连接
        std::string ip = "192.168.2.160";
        std::string local_ip = "192.168.2.2";
        // 错误码
        std::error_code ec;
        // 创建机器人对象，并且实例化机器人
        rokae::xMateRobot SDU_SR4;
        // 连接机器人
        SDU_SR4.connectToRobot(ip,local_ip);
        // 连接成功打印
        std::cout<<"机器人连接成功"<<std::endl;

        // 使用电脑控制需要自动操作模式
        SDU_SR4.setOperateMode(rokae::OperateMode::automatic,ec);
        if(ec){
            std::cerr<<"设置操作模式失败，失败原因："<<ec.message()<<std::endl;
            return -1;
        }

        // 设置为实时模式
        SDU_SR4.setMotionControlMode(rokae::MotionControlMode::RtCommand,ec);
        if(ec){
            std::cerr<<"设置控制模式失败，失败原因："<<ec.message()<<std::endl;
            return -1;
        }

        // 设置发送实时运动指令网络延迟阈值，超过阈值报警
        SDU_SR4.setRtNetworkTolerance(20,ec);

        // 上电
        SDU_SR4.setPowerState(true,ec);
        if(ec){
            std::cerr<<"上电失败，失败原因："<<ec.message()<<std::endl;
            return -1;
        }

        // 实例化专门的实时通道
        auto rtCon = SDU_SR4.getRtMotionController().lock();
        if(!rtCon){
            std::cerr<<"获取实时控制器失败"<<std::endl;
            return -1;
        }

        // 先低速运动到轨迹起点，再开始力矩跟踪。
        std::array<double, 6> q_start{};
        std::copy_n(trajectory.front().data(), 6, q_start.begin());
        rtCon->MoveJ(0.1, SDU_SR4.jointPos(ec), q_start);

        // 电脑需要接收什么信息
        SDU_SR4.startReceiveRobotState(std::chrono::milliseconds(1),
            {RtSupportedFields::jointPos_m,
             RtSupportedFields::jointVel_m,
             RtSupportedFields::motorTau});

        Torque cmd(6);
        std::array<double, 6> q{}, dq{}, tau_measured{};
        int step = 0;
        bool finished = false;
        std::mutex data_mutex;
        Eigen::VectorXd measured_state(12);
        std::vector<LogRow> logs(trajectory.size());
        int recorded_count = 0;

        // 主线程计划每control_dt秒计算一次MPC，预测参考点使用相同间隔。
        auto compute_torque = [&](int reference_step) {
            for (int i = 0; i <= N; ++i) {
                state_ref[i] = trajectory[std::min(reference_step + i * reference_stride, last)];
            }
            for (int i = 0; i < N; ++i) {
                ddq_ref[i] = acceleration[std::min(reference_step + i * reference_stride, last)];
            }

            // 只测量MPC计算耗时，不包含状态读取和指令发送。
            const auto solve_start = std::chrono::steady_clock::now();

            //-----------------------------------------------------------
            // 开始控制器的计算
            Eigen::VectorXd tau = controller.compute_control(current_state, state_ref, ddq_ref);
            //-----------------------------------------------------------

            const double solve_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - solve_start).count();

            ++solve_count;
            total_solve_ms += solve_ms;
            max_solve_ms = std::max(max_solve_ms, solve_ms);
            if (solve_ms > control_dt * 1000.0) ++overrun_count;

            // 首次及每10次打印一次；control_dt=0.05时，正常约每0.5秒打印一次。
            if (solve_count == 1 || solve_count % 10 == 0) {
                std::cout << "MPC第" << solve_count << "次计算：" << solve_ms
                          << " ms，平均：" << total_solve_ms / solve_count
                          << " ms，最大：" << max_solve_ms
                          << " ms，超过" << control_dt * 1000.0
                          << "ms：" << overrun_count << "次\n";
            }

            // MPC已输出不含重力的附加力矩，直接发送；重力由SDK补偿。
            return tau;
        };

        // 先算好初始力矩，再启动实时发送。
        SDU_SR4.getStateData(RtSupportedFields::jointPos_m, q);
        SDU_SR4.getStateData(RtSupportedFields::jointVel_m, dq);
        // 将状态拼接成12维的向量
        for (int j = 0; j < 6; ++j) {
            current_state(j) = q[j];
            current_state(j + 6) = dq[j];
        }
        measured_state = current_state;
        const Eigen::VectorXd initial_tau = compute_torque(0);
        // 保存第一次完整MPC求解的结果，启动SDK循环后再发送。
        std::copy_n(initial_tau.data(), 6, cmd.tau.begin());

        // 1. SDK线程：每1ms读取状态、记录数据，并返回最近一次算好的力矩。
        std::function<Torque(void)> callback = [&]() {
            SDU_SR4.getStateData(RtSupportedFields::jointPos_m, q);
            SDU_SR4.getStateData(RtSupportedFields::jointVel_m, dq);
            SDU_SR4.getStateData(RtSupportedFields::motorTau, tau_measured);
            std::lock_guard<std::mutex> lock(data_mutex);
            // 锁内更新共享状态；主线程读取时不会读到更新了一半的数据。
            for (int j = 0; j < 6; ++j) {
                measured_state(j) = q[j];
                measured_state(j + 6) = dq[j];
            }
            if (step <= last) {
                logs[step].index = step;
                logs[step].position = q;
                logs[step].velocity = dq;
                logs[step].torque = tau_measured;
                recorded_count = step + 1;
            }
            if (step == last) {
                cmd.setFinished();
                finished = true;
            }
            ++step; // 本次采样完成；下一次回调使用下一个编号。
            return cmd;
        };

        // 2. 正常结束和异常退出都使用相同的停止步骤。
        const auto stop_control = [&]() {
            rtCon->stopLoop(); // 先停止回调，再结束运动和状态接收。
            rtCon->stopMove();
            SDU_SR4.stopReceiveRobotState();
        };

        rtCon->setControlLoop(callback, 0, true); // 每次回调前自动更新关节状态
        try {
            // Clock只是steady_clock的短名字，使用不受系统时间调整影响的时钟。
            using Clock = std::chrono::steady_clock;
            const auto period = std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(control_dt));
            rtCon->startMove(RtControllerMode::torque);
            const auto start_time = Clock::now();
            const auto finish_time = start_time + std::chrono::milliseconds(last + 1);
            auto next_solve = start_time + period;
            rtCon->startLoop(false); // SDK后台发送，主线程计算MPC

            // 3. 主线程：等待 -> 复制状态 -> 计算MPC -> 更新力矩。
            while (Clock::now() < finish_time) {
                std::this_thread::sleep_until(std::min(next_solve, finish_time));
                if (Clock::now() >= finish_time) break;

                int reference_step;
                {
                    std::lock_guard<std::mutex> lock(data_mutex);
                    if (finished) break;
                    current_state = measured_state;
                    reference_step = std::max(0, step - 1);
                } // 离开花括号就释放锁，SDK可以继续更新状态和发送旧力矩。

                const Eigen::VectorXd tau = compute_torque(reference_step);

                {
                    std::lock_guard<std::mutex> lock(data_mutex);
                    std::copy_n(tau.data(), 6, cmd.tau.begin());
                } // 发布新力矩，之后的SDK回调会取到它。

                // 求解超时时跳过错过的周期，不连续补算。
                do {
                    next_solve += period;
                } while (next_solve < Clock::now());
            }
        } catch (...) {
            stop_control();
            throw; // 清理后，把错误交给外层catch打印。
        }
        stop_control();

        // 4. 正常结束后统一写文件，避免文件操作影响1ms实时发送线程。
        writeLog(output_txt, logs, recorded_count);

        std::cout << "力矩控制结束，实际关节状态已保存到："
                  << output_txt << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr <<"MPC控制失败："<< e.what() << '\n';
        exit_code = -1;
    }

    // 正常结束或SDK异常退出时，均输出已完成的MPC计算统计。
    if (solve_count > 0) {
        std::cout << "MPC耗时统计：计算" << solve_count
                  << "次，平均：" << total_solve_ms / solve_count
                  << " ms，最大：" << max_solve_ms
                  << " ms，超过" << control_dt * 1000.0
                  << "ms：" << overrun_count << "次" << std::endl;
    }
    return exit_code;
    
}
