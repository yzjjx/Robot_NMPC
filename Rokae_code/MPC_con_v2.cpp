// 读取圆轨迹：测量关节状态 -> MPC计算力矩 -> 发送力矩。
#include "NMPC_control.h"
#include "pinocchio_fun.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
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
using Clock = std::chrono::steady_clock;

struct LogRow
{
    int index = 0;
    double elapsed_ms = 0.0;
    int reference_index = 0;
    std::array<double, 6> position{};
    std::array<double, 6> velocity{};
    std::array<double, 6> torque{};
    std::array<double, 6> commanded_torque{};
};

void writeLog(const std::string& filename,
              const std::vector<LogRow>& logs,
              int recorded_count)
{
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("无法创建实际关节状态文件");
    }

    file << "index,elapsed_ms,reference_index,vel,pos,tau,tau_cmd\n";
    file << std::setprecision(10);
    for (int row_index = 0; row_index < recorded_count; ++row_index) {
        const LogRow& row = logs[row_index];
        file << row.index << "," << row.elapsed_ms << "," << row.reference_index << ",";

        const auto write_array = [&](const std::array<double, 6>& values) {
            file << "\"[";
            for (int joint = 0; joint < 6; ++joint) {
                if (joint > 0) file << ",";
                file << values[joint];
            }
            file << "]\"";
        };

        write_array(row.velocity);
        file << ",";
        write_array(row.position);
        file << ",";
        write_array(row.torque);
        file << ",";
        write_array(row.commanded_torque);
        file << "\n";
    }
}

int main()
{
    const double control_dt = 0.02; // MPC控制周期：10ms，即100Hz
    const double sample_dt = 0.001; // 轨迹采样周期和SDK发送周期：1ms
    // 主线程在求解前后检查回调心跳；100ms为诊断阈值，不是硬实时看门狗。
    const double callback_timeout_ms = 100.0;

    // 在输入的期望轨迹中隔着reference_stride行取一个参考点
    const int reference_stride = static_cast<int>(std::lround(control_dt / sample_dt));
    int solve_count = 0;
    int overrun_count = 0;
    double total_solve_ms = 0.0;
    double max_solve_ms = 0.0;
    int exit_code = 0;
    const std::string input_q = "../data_in/circle_R400_joint_poses_SR4_V50.txt";
    const std::string output_txt = "../data_out/circle_R400_SR4.txt";

    try
    {
        // 文件每行：时间(s)、6个关节位置(rad)、6个关节速度(rad/s)。
        std::ifstream file(input_q);
        if (!file) {
            throw std::runtime_error("无法打开轨迹文件：" + input_q);
        }
        std::string header;
        std::getline(file, header);
        std::vector<mpc_v2::State> trajectory;
        double time;
        while (file >> time) {
            mpc_v2::State state;
            for (int j = 0; j < mpc_v2::NX; ++j) {
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
        std::vector<mpc_v2::Joint> acceleration(trajectory.size());
        for (int i = 0; i < last; ++i) {
            acceleration[i] =
                (trajectory[i + 1].tail<mpc_v2::DOF>() -
                 trajectory[i].tail<mpc_v2::DOF>()) / sample_dt;
        }
        acceleration[last] = mpc_v2::Joint::Zero();

        //----------------------------------------------------------------
        // 输入动力学模型和控制器
        mpc_v2::Dynamics dynamics("../urdf/ROKAE_SR4.urdf");
        mpc_v2::Controller controller(dynamics, control_dt, 30);
        //----------------------------------------------------------------

        const int N = controller.horizon();
        Eigen::MatrixXd state_ref = Eigen::MatrixXd::Zero(N + 1, mpc_v2::NX);
        Eigen::MatrixXd ddq_ref = Eigen::MatrixXd::Zero(N, mpc_v2::DOF);
        mpc_v2::State current_state = mpc_v2::State::Zero();

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

        // SDK要求在非实时模式下设置丢包阈值；先退出上次可能遗留的实时模式。
        SDU_SR4.setMotionControlMode(rokae::MotionControlMode::NrtCommand,ec);
        if(ec){
            throw std::runtime_error("切换非实时模式失败：" + ec.message());
        }
        SDU_SR4.setRtNetworkTolerance(20,ec);
        if(ec){
            throw std::runtime_error("设置实时网络容差失败：" + ec.message());
        }

        // 设置为实时模式
        SDU_SR4.setMotionControlMode(rokae::MotionControlMode::RtCommand,ec);
        if(ec){
            std::cerr<<"设置控制模式失败，失败原因："<<ec.message()<<std::endl;
            return -1;
        }

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
        mpc_v2::State measured_state = mpc_v2::State::Zero();
        std::vector<LogRow> logs(trajectory.size());
        int recorded_count = 0;
        Clock::time_point start_time{};
        Clock::time_point last_state_update{};
        int command_reference_step = 0;
        std::exception_ptr callback_error;

        // 主线程计划每control_dt秒计算一次MPC，预测参考点使用相同间隔。
        auto compute_torque = [&](int reference_step) {
            for (int i = 0; i <= N; ++i) {
                state_ref.row(i) =
                    trajectory[std::min(reference_step + i * reference_stride, last)].transpose();
            }
            for (int i = 0; i < N; ++i) {
                ddq_ref.row(i) =
                    acceleration[std::min(reference_step + i * reference_stride, last)].transpose();
            }

            // 只测量MPC计算耗时，不包含状态读取和指令发送。
            const auto solve_start = std::chrono::steady_clock::now();

            //-----------------------------------------------------------
            // 开始控制器的计算
            mpc_v2::Joint tau = controller.compute_control(current_state, state_ref, ddq_ref);
            //-----------------------------------------------------------

            const double solve_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - solve_start).count();

            ++solve_count;
            total_solve_ms += solve_ms;
            max_solve_ms = std::max(max_solve_ms, solve_ms);
            if (solve_ms > control_dt * 1000.0) ++overrun_count;

            // 首次及每10次打印一次；正常约每0.1秒打印一次。
            if (solve_count == 1 || solve_count % 10 == 0) {
                std::cout << "MPC第" << solve_count << "次计算：" << solve_ms
                          << " ms，平均：" << total_solve_ms / solve_count
                          << " ms，最大：" << max_solve_ms
                          << " ms，超过" << control_dt * 1000.0
                          << "ms：" << overrun_count << "次"
                          << "，参考点=" << reference_step << "/" << last
                          << "，|dq|=" << current_state.tail<6>().norm()
                          << " rad/s，最大位置误差="
                          << (current_state.head<6>() - trajectory[reference_step].head<6>()).cwiseAbs().maxCoeff()
                          << " rad，tau_mpc=[" << tau.transpose() << "] N*m"
                          << std::endl;
            }

            // MPC已输出不含重力的附加力矩，直接发送；重力由SDK补偿。
            return tau;
        };

        // 1. SDK线程：每1ms读取状态、记录数据，并返回最近一次算好的力矩。
        std::function<Torque(void)> callback = [&]() {
            try {
                if (SDU_SR4.getStateData(RtSupportedFields::jointPos_m, q) != 0 ||
                    SDU_SR4.getStateData(RtSupportedFields::jointVel_m, dq) != 0 ||
                    SDU_SR4.getStateData(RtSupportedFields::motorTau, tau_measured) != 0) {
                    throw std::runtime_error("SDK回调读取关节状态失败");
                }
                std::lock_guard<std::mutex> lock(data_mutex);
                last_state_update = Clock::now();
                // 锁内更新共享状态；主线程读取时不会读到更新了一半的数据。
                for (int j = 0; j < 6; ++j) {
                    measured_state(j) = q[j];
                    measured_state(j + 6) = dq[j];
                }
                if (step <= last) {
                    logs[step].index = step;
                    logs[step].elapsed_ms = std::chrono::duration<double, std::milli>(
                        last_state_update - start_time).count();
                    logs[step].reference_index = command_reference_step;
                    logs[step].position = q;
                    logs[step].velocity = dq;
                    logs[step].torque = tau_measured;
                    std::copy_n(cmd.tau.begin(), 6, logs[step].commanded_torque.begin());
                    recorded_count = step + 1;
                }
                if (step == last) {
                    cmd.setFinished();
                    finished = true;
                }
                ++step; // 本次采样完成；下一次回调使用下一个编号。
                return cmd;
            } catch (...) {
                // 回调内不打印；主线程读取异常，SDK仍按自身机制处理该异常。
                std::lock_guard<std::mutex> lock(data_mutex);
                callback_error = std::current_exception();
                throw;
            }
        };

        // 2. 正常结束和异常退出都使用相同的停止步骤。
        const auto stop_control = [&]() {
            std::exception_ptr first_error;
            const auto attempt = [&](const char* name, auto action) {
                std::cerr << "停止步骤：" << name << std::endl;
                try {
                    action();
                    std::cerr << name << "完成" << std::endl;
                } catch (const std::exception& e) {
                    if (!first_error) first_error = std::current_exception();
                    std::cerr << name << "异常：" << e.what() << std::endl;
                } catch (...) {
                    if (!first_error) first_error = std::current_exception();
                    std::cerr << name << "发生未知异常" << std::endl;
                }
            };
            // stopLoop可能重新抛出后台异常；仍尝试后续停止步骤。
            attempt("stopLoop", [&]() { rtCon->stopLoop(); });
            attempt("stopMove", [&]() { rtCon->stopMove(); });
            attempt("stopReceiveRobotState", [&]() { SDU_SR4.stopReceiveRobotState(); });
            return first_error;
        };

        rtCon->setControlLoop(callback, 0, true); // 每次回调前自动更新关节状态
        std::exception_ptr control_error;
        std::string stop_reason = "达到轨迹时间上限";
        try {
            // 初始求解也纳入异常清理范围。
            if (SDU_SR4.getStateData(RtSupportedFields::jointPos_m, q) != 0 ||
                SDU_SR4.getStateData(RtSupportedFields::jointVel_m, dq) != 0) {
                throw std::runtime_error("读取初始关节状态失败");
            }
            for (int j = 0; j < 6; ++j) {
                current_state(j) = q[j];
                current_state(j + 6) = dq[j];
            }
            measured_state = current_state;
            const mpc_v2::Joint initial_tau = compute_torque(0);
            std::copy_n(initial_tau.data(), 6, cmd.tau.begin());

            // Clock只是steady_clock的短名字，使用不受系统时间调整影响的时钟。
            const auto period = std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(control_dt));
            rtCon->startMove(RtControllerMode::torque);
            start_time = Clock::now();
            last_state_update = start_time;
            const auto finish_time = start_time + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(trajectory.size() * sample_dt));
            auto next_solve = start_time + period;
            rtCon->startLoop(false); // SDK后台发送，主线程计算MPC

            // 调用时持有data_mutex；心跳代表回调成功读取了状态，不代表实物在运动。
            const auto check_callback = [&]() {
                if (callback_error) std::rethrow_exception(callback_error);
                if (finished) return;
                const double age_ms = std::chrono::duration<double, std::milli>(
                    Clock::now() - last_state_update).count();
                if (age_ms > callback_timeout_ms) {
                    throw std::runtime_error("SDK回调超过" + std::to_string(age_ms) +
                        "ms未更新，回调次数=" + std::to_string(step) +
                        "，最后参考点=" + std::to_string(std::max(0, step - 1)) +
                        "/" + std::to_string(last));
                }
            };

            // 3. 主线程：等待 -> 复制状态 -> 计算MPC -> 更新力矩。
            while (Clock::now() < finish_time) {
                std::this_thread::sleep_until(std::min(next_solve, finish_time));
                if (Clock::now() >= finish_time) break;

                int reference_step;
                int callback_count;
                double state_age_ms;
                {
                    std::lock_guard<std::mutex> lock(data_mutex);
                    check_callback();
                    if (finished) {
                        stop_reason = "SDK回调已遍历全部轨迹点";
                        break;
                    }
                    // 首帧到来前仅等待；超时由check_callback报告。
                    if (step == 0) {
                        next_solve = Clock::now() + period;
                        continue;
                    }
                    current_state = measured_state;
                    reference_step = std::max(0, step - 1);
                    callback_count = step;
                    state_age_ms = std::chrono::duration<double, std::milli>(
                        Clock::now() - last_state_update).count();
                } // 离开花括号就释放锁，SDK可以继续更新状态和发送旧力矩。

                if (rtCon->hasMotionError()) {
                    throw std::runtime_error("SDK hasMotionError=true，参考点=" +
                        std::to_string(reference_step) + "/" + std::to_string(last));
                }
                const mpc_v2::Joint tau = compute_torque(reference_step);

                if (solve_count % 10 == 0) {
                    std::cout << "回调诊断：次数=" << callback_count
                              << "，求解前状态年龄=" << state_age_ms << " ms"
                              << "，运行时间=" << std::chrono::duration<double>(Clock::now() - start_time).count()
                              << " s" << std::endl;
                }
                if (rtCon->hasMotionError()) {
                    throw std::runtime_error("MPC求解期间SDK报告运动错误");
                }
                {
                    std::lock_guard<std::mutex> lock(data_mutex);
                    check_callback();
                    if (finished) {
                        stop_reason = "SDK回调已遍历全部轨迹点";
                        break;
                    }
                    if (Clock::now() >= finish_time) break;
                    std::copy_n(tau.data(), 6, cmd.tau.begin());
                    command_reference_step = reference_step;
                } // 发布新力矩，之后的SDK回调会取到它。

                // 求解超时时跳过错过的周期，不连续补算。
                do {
                    next_solve += period;
                } while (next_solve < Clock::now());
            }
        } catch (...) {
            control_error = std::current_exception();
            stop_reason = "控制异常";
            // 先报告原始原因，避免SDK停止过程阻塞时终端没有任何提示。
            try {
                std::rethrow_exception(control_error);
            } catch (const std::exception& e) {
                std::cerr << "控制中断：" << e.what() << std::endl;
            } catch (...) {
                std::cerr << "控制中断：未知异常" << std::endl;
            }
        }
        std::cerr << "开始停止控制，原因：" << stop_reason << std::endl;
        int final_callback_count;
        double final_callback_age_ms;
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            final_callback_count = step;
            final_callback_age_ms = step > 0 ? std::chrono::duration<double, std::milli>(
                Clock::now() - last_state_update).count() : -1.0;
        }
        std::cerr << "停止前回调次数=" << final_callback_count << "/" << trajectory.size()
                  << "，距上次状态更新=" << final_callback_age_ms << " ms（-1表示无回调）"
                  << std::endl;
        const auto cleanup_error = stop_control();
        if (!control_error) control_error = cleanup_error;

        // 4. 正常和异常退出都保存已记录的数据，写文件不占用SDK回调线程。
        int saved_count;
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            saved_count = recorded_count;
        }
        std::cout << "停止时已记录 " << saved_count << "/" << trajectory.size()
                  << " 个轨迹采样点" << std::endl;
        try {
            writeLog(output_txt, logs, saved_count);
            std::cout << "实际关节状态已保存到：" << output_txt << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "保存状态失败：" << e.what() << std::endl;
            if (!control_error) control_error = std::current_exception();
        }

        if (control_error || saved_count < static_cast<int>(trajectory.size())) {
            // 查询机器人侧日志仅放在控制停止后，不占用正常控制周期。
            std::error_code log_ec;
            const auto controller_logs = SDU_SR4.queryControllerLog(10, {}, log_ec);
            if (log_ec) {
                std::cerr << "查询控制器日志失败：" << log_ec.message() << std::endl;
            } else {
                std::cerr << "控制器最近日志（可能包含本次运行之前的记录）：" << std::endl;
                for (const auto& entry : controller_logs) {
                    std::cerr << entry.timestamp << " [" << entry.id << "] "
                              << entry.content << " " << entry.repair << std::endl;
                }
            }
        }
        if (control_error) std::rethrow_exception(control_error);
    }
    catch(const std::exception& e)
    {
        std::cerr <<"MPC控制失败："<< e.what() << '\n';
        exit_code = -1;
    }
    catch (...) {
        std::cerr << "MPC控制失败：未知异常" << std::endl;
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
