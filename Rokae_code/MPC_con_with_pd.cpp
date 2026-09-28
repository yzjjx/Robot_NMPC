// MPC每50ms更新一次，关节PD每1ms更新一次。
// 发送力矩 = 限幅(MPC力矩 + PD力矩)，重力补偿由机器人底层完成。
// 本例在原MPC输出上叠加PD，没有把PD反馈加入MPC内部的预测模型。
#include "NMPC_control.h"
#include "pinocchio_fun.h"
#include "rokae/robot.h"

#include <algorithm>
#include <array>
#include <chrono>
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

using namespace rokae;

// PD是一个普通函数：输入实际状态、期望状态和增益，返回六个关节的修正力矩。
// q、dq：实际角度(rad)、实际速度(rad/s)。
// state_ref：前6项是期望角度，后6项是期望速度。
// kp：位置误差增益，单位N*m/rad；kd：速度误差增益，单位N*m*s/rad。
std::array<double, 6> compute_pd(
    const std::array<double, 6>& q,
    const std::array<double, 6>& dq,
    const Eigen::VectorXd& state_ref,
    const std::array<double, 6>& kp,
    const std::array<double, 6>& kd)
{
    std::array<double, 6> tau_pd{};
    for (int j = 0; j < 6; ++j) {
        const double position_error = state_ref(j) - q[j];
        const double velocity_error = state_ref(j + 6) - dq[j];
        tau_pd[j] = kp[j] * position_error + kd[j] * velocity_error;
    }
    return tau_pd;
}

struct LogRow
{
    int index = 0;
    std::array<double, 6> position{};
    std::array<double, 6> velocity{};
    std::array<double, 6> measured_torque{};
    std::array<double, 6> mpc_torque{};
    std::array<double, 6> pd_torque{};
    std::array<double, 6> command_torque{};
};

// 正常停止控制后写CSV，每个关节占一列，方便比较MPC、PD和最终发送力矩。
void writeLog(const std::string& filename,
              const std::vector<LogRow>& logs, int recorded_count)
{
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("无法创建日志文件：" + filename);

    file << "index";
    for (const std::string& name : {"q_rad", "dq_rad_s", "tau_measured_Nm",
                                   "tau_mpc_Nm", "tau_pd_Nm", "tau_cmd_Nm"}) {
        for (int j = 1; j <= 6; ++j) file << "," << name << "_" << j;
    }
    file << '\n' << std::setprecision(10);

    const auto write_array = [&](const std::array<double, 6>& values) {
        for (double value : values) file << "," << value;
    };
    for (int i = 0; i < recorded_count; ++i) {
        const LogRow& row = logs[i];
        file << row.index;
        write_array(row.position);
        write_array(row.velocity);
        write_array(row.measured_torque);
        write_array(row.mpc_torque);
        write_array(row.pd_torque);
        write_array(row.command_torque);
        file << '\n';
    }
}

int main()
{
    const double control_dt = 0.05; // MPC周期：50ms
    const double sample_dt = 0.001; // 轨迹采样、SDK回调和PD周期：1ms
    const int reference_stride = static_cast<int>(std::lround(control_dt / sample_dt));

    // 每个数对应一个关节。这些是教学示例值，尚未经过实机整定。
    // kp、kd全部设为0时，PD修正为0，恢复仅使用MPC力矩。
    const std::array<double, 6> kp = {10, 10, 10, 5, 5, 5};
    const std::array<double, 6> kd = {1, 1, 1, 0.5, 0.5, 0.5};
    // 延用原MPC的±30 N*m附加力矩范围；限制叠加后的总指令。
    // 这是软件限幅，不代表各关节的额定力矩。
    const double torque_limit = 30.0;

    const std::string input_q = "../data_in/circle_R200_joint_trajectory_SR4_V50.txt";
    const std::string output_csv = "../data_out/circle_R200_SR4_with_pd.csv";
    int solve_count = 0;
    int overrun_count = 0;
    double total_solve_ms = 0.0;
    double max_solve_ms = 0.0;
    int exit_code = 0;

    try {
        // 1. 读取轨迹：每行是时间、6个角度、6个速度；第一行是表头。
        std::ifstream file(input_q);
        if (!file) throw std::runtime_error("无法打开轨迹文件：" + input_q);
        std::string header;
        std::getline(file, header);
        std::vector<Eigen::VectorXd> trajectory;
        double time;
        while (file >> time) {
            Eigen::VectorXd state(12);
            for (int j = 0; j < 12; ++j) file >> state(j);
            if (!file || !std::isfinite(time) || !state.allFinite()) {
                throw std::runtime_error("轨迹数据不完整或包含无效数值");
            }
            trajectory.push_back(state);
        }
        if (trajectory.size() < 2) throw std::runtime_error("轨迹至少需要两个点");

        // 与原程序一样，后续按sample_dt安排参考点，文件采样间隔需为1ms。
        const int last = static_cast<int>(trajectory.size()) - 1;
        std::vector<Eigen::VectorXd> acceleration(trajectory.size());
        for (int i = 0; i < last; ++i) {
            acceleration[i] = (trajectory[i + 1].tail(6) - trajectory[i].tail(6)) / sample_dt;
        }
        acceleration[last] = Eigen::VectorXd::Zero(6);

        // 2. 创建原来的MPC控制器，向前预测14个区间，即0.7秒。
        pinocchioFun dynamics("../urdf/ROKAE_SR4.urdf");
        ROKAE_NMPC controller(dynamics, control_dt, 14);
        const int N = controller.horizon();
        std::vector<Eigen::VectorXd> state_ref(N + 1, Eigen::VectorXd::Zero(12));
        std::vector<Eigen::VectorXd> ddq_ref(N, Eigen::VectorXd::Zero(6));
        Eigen::VectorXd current_state(12);

        // 只计算MPC力矩。PD在后面的SDK回调中，根据最新状态单独计算。
        const auto compute_mpc = [&](int reference_step) {
            for (int i = 0; i <= N; ++i) {
                state_ref[i] = trajectory[std::min(reference_step + i * reference_stride, last)];
            }
            for (int i = 0; i < N; ++i) {
                ddq_ref[i] = acceleration[std::min(reference_step + i * reference_stride, last)];
            }

            const auto begin = std::chrono::steady_clock::now();
            Eigen::VectorXd tau = controller.compute_control(current_state, state_ref, ddq_ref);
            if (!controller.qp_success() || !tau.allFinite()) {
                throw std::runtime_error("MPC求解失败或输出无效力矩");
            }
            const double solve_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
            ++solve_count;
            total_solve_ms += solve_ms;
            max_solve_ms = std::max(max_solve_ms, solve_ms);
            if (solve_ms > control_dt * 1000.0) ++overrun_count;
            if (solve_count == 1 || solve_count % 10 == 0) {
                std::cout << "MPC第" << solve_count << "次计算：" << solve_ms << " ms\n";
            }
            return tau;
        };

        // 3. 连接机器人，设置模式，并移动到轨迹起点。
        std::error_code ec;
        xMateRobot robot;
        robot.connectToRobot("192.168.2.160", "192.168.2.2");
        std::cout << "机器人连接成功\n";
        robot.setOperateMode(OperateMode::automatic, ec);
        if (ec) throw std::runtime_error("设置操作模式失败：" + ec.message());
        robot.setMotionControlMode(MotionControlMode::RtCommand, ec);
        if (ec) throw std::runtime_error("设置实时模式失败：" + ec.message());
        robot.setRtNetworkTolerance(20, ec);
        if (ec) throw std::runtime_error("设置网络容忍参数失败：" + ec.message());
        robot.setPowerState(true, ec);
        if (ec) throw std::runtime_error("上电失败：" + ec.message());
        auto rtCon = robot.getRtMotionController().lock();
        if (!rtCon) throw std::runtime_error("获取实时控制器失败");
        std::array<double, 6> q_start{};
        std::copy_n(trajectory.front().data(), 6, q_start.begin());
        const auto initial_position = robot.jointPos(ec);
        if (ec) throw std::runtime_error("读取初始位置失败：" + ec.message());
        rtCon->MoveJ(0.1, initial_position, q_start);

        // SDK线程写measured_state，主线程写tau_mpc；访问时使用同一把锁。
        std::mutex data_mutex;
        Eigen::VectorXd measured_state(12);
        Eigen::VectorXd tau_mpc = Eigen::VectorXd::Zero(6);
        std::array<double, 6> q{}, dq{}, tau_measured{};
        Torque cmd(6); // 启动后只由SDK回调填写最终指令。
        std::vector<LogRow> logs(trajectory.size());
        int step = 0;
        int recorded_count = 0;
        bool finished = false;
        bool invalid_feedback = false;

        robot.startReceiveRobotState(std::chrono::milliseconds(1),
            {RtSupportedFields::jointPos_m, RtSupportedFields::jointVel_m,
             RtSupportedFields::motorTau});

        // 先算好初始MPC力矩；失败时，此时还没有启动后台控制循环。
        try {
            robot.getStateData(RtSupportedFields::jointPos_m, q);
            robot.getStateData(RtSupportedFields::jointVel_m, dq);
            for (int j = 0; j < 6; ++j) {
                current_state(j) = q[j];
                current_state(j + 6) = dq[j];
            }
            measured_state = current_state;
            tau_mpc = compute_mpc(0);
        } catch (...) {
            robot.stopReceiveRobotState();
            throw;
        }

        // 4. SDK每1ms调用一次：读状态 -> 算PD -> 叠加限幅 -> 记录 -> 返回指令。
        std::function<Torque(void)> callback = [&]() {
            robot.getStateData(RtSupportedFields::jointPos_m, q);
            robot.getStateData(RtSupportedFields::jointVel_m, dq);
            robot.getStateData(RtSupportedFields::motorTau, tau_measured);
            std::lock_guard<std::mutex> lock(data_mutex);
            if (finished) return cmd;

            for (int j = 0; j < 6; ++j) {
                measured_state(j) = q[j];
                measured_state(j + 6) = dq[j];
            }
            // PD使用当前采样点，不能使用主线程可能正在更新的state_ref。
            const std::array<double, 6> tau_pd = compute_pd(q, dq, trajectory[step], kp, kd);
            for (int j = 0; j < 6; ++j) {
                const double total_tau = tau_mpc(j) + tau_pd[j];
                if (!std::isfinite(q[j]) || !std::isfinite(dq[j]) || !std::isfinite(total_tau)) {
                    // 无效反馈时不发送NaN/Inf；通知主线程停止并报告错误。
                    std::fill(cmd.tau.begin(), cmd.tau.end(), 0.0);
                    cmd.setFinished();
                    finished = true;
                    invalid_feedback = true;
                    return cmd;
                }
                // 每次都从MPC力矩重新叠加，不能写成cmd.tau[j] += tau_pd[j]。
                cmd.tau[j] = std::clamp(total_tau, -torque_limit, torque_limit);
            }

            LogRow& row = logs[step];
            row.index = step;
            row.position = q;
            row.velocity = dq;
            row.measured_torque = tau_measured;
            row.pd_torque = tau_pd;
            for (int j = 0; j < 6; ++j) {
                row.mpc_torque[j] = tau_mpc(j);
                row.command_torque[j] = cmd.tau[j];
            }
            recorded_count = step + 1;
            if (step == last) {
                cmd.setFinished();
                finished = true;
            }
            ++step;
            return cmd;
        };

        const auto stop_control = [&]() {
            rtCon->stopLoop();
            rtCon->stopMove();
            robot.stopReceiveRobotState();
        };

        // 5. 主线程每50ms计算一次MPC。计算期间，SDK继续计算PD并发送力矩。
        try {
            rtCon->setControlLoop(callback, 0, true);
            rtCon->startMove(RtControllerMode::torque);
            using Clock = std::chrono::steady_clock;
            const auto period = std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(control_dt));
            const auto start_time = Clock::now();
            const auto finish_time = start_time + std::chrono::milliseconds(last + 1);
            auto next_solve = start_time + period;
            rtCon->startLoop(false);

            while (Clock::now() < finish_time) {
                std::this_thread::sleep_until(std::min(next_solve, finish_time));
                if (Clock::now() >= finish_time) break;
                int reference_step;
                {
                    std::lock_guard<std::mutex> lock(data_mutex);
                    if (finished) break;
                    current_state = measured_state;
                    reference_step = std::max(0, step - 1);
                }
                // 耗时的MPC计算在锁外进行，避免一直占着锁。
                const Eigen::VectorXd new_tau_mpc = compute_mpc(reference_step);
                {
                    std::lock_guard<std::mutex> lock(data_mutex);
                    if (finished) break;
                    tau_mpc = new_tau_mpc; // 只更新MPC分量，PD由SDK线程单独计算。
                }
                do {
                    next_solve += period;
                } while (next_solve < Clock::now());
            }
        } catch (...) {
            stop_control();
            throw;
        }
        stop_control();

        // 6. 后台线程已停止，统一保存日志。
        writeLog(output_csv, logs, recorded_count);
        if (invalid_feedback) throw std::runtime_error("PD反馈或叠加力矩出现无效数值，控制已停止");
        std::cout << "MPC+PD控制结束，记录已保存到：" << output_csv << '\n';
    } catch (const std::exception& error) {
        std::cerr << "MPC+PD控制失败：" << error.what() << '\n';
        exit_code = -1;
    }

    if (solve_count > 0) {
        std::cout << "MPC求解次数：" << solve_count
                  << "，平均耗时：" << total_solve_ms / solve_count
                  << " ms，最大耗时：" << max_solve_ms
                  << " ms，超过" << control_dt * 1000.0
                  << "ms：" << overrun_count << "次\n";
    }
    return exit_code;
}
