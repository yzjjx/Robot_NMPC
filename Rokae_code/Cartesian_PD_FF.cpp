// 笛卡尔 PD + 关节力矩前馈，从 build 目录运行。
// tau_sdk = [RNEA(q_ref, dq_ref, ddq_ref) - g(q_ref)]
//         + J(q)^T [Kp * e_pose + Kd * (V_ref - J(q) * dq)]
// SDK 已补偿实际机器人的重力和摩擦，保持其默认配置，不重复补偿。
// 前馈在参考状态求值，所以减去同一参考状态的 g(q_ref)，而非 g(q)。
// 位姿误差、速度和雅可比均使用末端原点、世界轴方向：LOCAL_WORLD_ALIGNED。

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <Eigen/Dense>
#include "rokae/robot.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
constexpr double sample_dt = 0.001;
volatile std::sig_atomic_t stop_requested = 0;

void requestStop(int) { stop_requested = 1; }

struct Reference {
    double time = 0.0;
    Vec6 q = Vec6::Zero();
    Vec6 dq = Vec6::Zero();
    Vec6 ddq = Vec6::Zero();
    pinocchio::SE3 pose = pinocchio::SE3::Identity();
    Vec6 velocity = Vec6::Zero(); // 前三维线速度，后三维角速度
    Vec6 tau_ff = Vec6::Zero();
};

struct LogRow {
    double time = 0.0;
    double wall_time = 0.0;
    double compute_us = 0.0;
    Vec6 q = Vec6::Zero();
    Vec6 dq = Vec6::Zero();
    Vec6 pose_error = Vec6::Zero();
    Vec6 tau_ff = Vec6::Zero();
    Vec6 tau_pd = Vec6::Zero();
    Vec6 tau_cmd = Vec6::Zero();
    Vec6 tau_measured = Vec6::Zero(); // SDK motorTau，并非无重力指令
    bool limited = false;
};

std::vector<Reference> loadReference(const std::string& path,
                                    const pinocchio::Model& model,
                                    pinocchio::FrameIndex frame_id)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("无法打开轨迹文件：" + path);
    std::string line;
    std::getline(file, line); // time_s, q1...q6, dq1...dq6 的表头
    std::vector<Reference> refs;
    std::size_t line_number = 1;
    while (std::getline(file, line)) {
        ++line_number;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        Reference ref;
        std::istringstream row(line);
        row >> ref.time;
        for (int j = 0; j < 6; ++j) row >> ref.q[j];
        for (int j = 0; j < 6; ++j) row >> ref.dq[j];
        std::string extra;
        if (!row || (row >> extra) || !std::isfinite(ref.time)
            || !ref.q.allFinite() || !ref.dq.allFinite()) {
            throw std::runtime_error("轨迹行格式错误：" + std::to_string(line_number));
        }
        if (!refs.empty()
            && std::abs(ref.time - refs.back().time - sample_dt) > 1e-7) {
            throw std::runtime_error("输入轨迹时间戳必须严格按 1ms 递增");
        }
        if ((ref.q.array() < model.lowerPositionLimit.array()).any()
            || (ref.q.array() > model.upperPositionLimit.array()).any()
            || (ref.dq.cwiseAbs().array() > model.velocityLimit.array()).any()) {
            throw std::runtime_error("参考状态超过 URDF 限位，行：" + std::to_string(line_number));
        }
        refs.push_back(ref);
    }
    if (file.bad()) throw std::runtime_error("读取轨迹时发生 I/O 错误");
    if (refs.size() < 3) throw std::runtime_error("轨迹至少需要三个采样点");
    // MoveJ 到起点后从静止启动力矩跟踪；终点也要求静止。
    if (refs.front().dq.norm() > 1e-3 || refs.back().dq.norm() > 1e-3) {
        throw std::runtime_error("轨迹起点和终点的关节速度应接近零");
    }

    pinocchio::Data data(model);
    Mat6 J = Mat6::Zero();
    const double time_origin = refs.front().time;
    for (std::size_t i = 0; i < refs.size(); ++i) {
        // 中心差分计算关节加速度，端点使用单边差分。
        const std::size_t left = (i == 0) ? 0 : i - 1;
        const std::size_t right = std::min(i + 1, refs.size() - 1);
        refs[i].ddq = (refs[right].dq - refs[left].dq)
                    / (static_cast<double>(right - left) * sample_dt);
        pinocchio::computeJointJacobians(model, data, refs[i].q);
        pinocchio::updateFramePlacements(model, data);
        J.setZero();
        pinocchio::getFrameJacobian(model, data, frame_id,
                                   pinocchio::LOCAL_WORLD_ALIGNED, J);
        refs[i].pose = data.oMf[frame_id];
        refs[i].velocity = J * refs[i].dq;
        // 拷贝 RNEA 返回值，防止后续算法修改 Data 内部缓存。
        const Vec6 tau_rnea = pinocchio::rnea(model, data, refs[i].q,
                                             refs[i].dq, refs[i].ddq);
        const Vec6 gravity = pinocchio::computeGeneralizedGravity(model, data, refs[i].q);
        refs[i].tau_ff = tau_rnea - gravity;
        refs[i].time -= time_origin;
        if (!refs[i].ddq.allFinite() || !refs[i].velocity.allFinite()
            || !refs[i].pose.toHomogeneousMatrix().allFinite()
            || !refs[i].tau_ff.allFinite()) {
            throw std::runtime_error("参考运动学或前馈力矩包含非有限值");
        }
    }
    return refs;
}

void writeLog(std::ofstream& file, const std::vector<LogRow>& logs, std::size_t count)
{
    file << "time_s,wall_time_s,compute_us,limited";
    for (const char* field : {"q", "dq", "pose_error", "tau_ff", "tau_pd", "tau_cmd", "motor_tau"}) {
        for (int j = 1; j <= 6; ++j) file << ',' << field << j;
    }
    file << '\n' << std::setprecision(12);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& r = logs[i];
        file << r.time << ',' << r.wall_time << ',' << r.compute_us << ',' << r.limited;
        for (const Vec6* values : {&r.q, &r.dq, &r.pose_error, &r.tau_ff,
                                  &r.tau_pd, &r.tau_cmd, &r.tau_measured}) {
            for (int j = 0; j < 6; ++j) file << ',' << (*values)[j];
        }
        file << '\n';
    }
    file.flush();
    if (!file) throw std::runtime_error("保存笛卡尔 PD 日志失败");
}

void checkError(const std::error_code& ec, const char* operation)
{
    if (ec) throw std::runtime_error(std::string(operation) + "：" + ec.message());
}
} // namespace

int main()
{
    const std::string input_q = "../data_in/circle_R200_joint_trajectory_SR4_V50.txt";
    const std::string urdf_path = "../urdf/ROKAE_SR4.urdf";
    const std::string output_txt = "../data_out/cartesian_pd_ff.csv";
    const std::string robot_ip = "192.168.2.160";
    const std::string local_ip = "192.168.2.2";
    // 控制点为此 URDF frame 的原点，不是 SDK 中任意配置的工具 TCP。
    // 使用工具时需在 URDF 中加入对应固定 frame 和负载惯量，并修改这里。
    const std::string end_frame = "xMateSR4C_link6";

    Vec6 kp, kd, tau_limit, tau_rate_limit;
    kp << 200, 200, 200, 5, 5, 5; // 平移 N/m；旋转 Nm/rad
    kd << 30, 30, 30, 1, 1, 1;    // 平移 Ns/m；旋转 Nms/rad
    // 以下为用户侧初始调试限幅，不是 SR4 的厂家额定限值。
    // 限制的是发给 SDK 的附加力矩，不包含 SDK 内部重力和摩擦补偿。
    tau_limit << 30, 30, 30, 30, 30, 30;            // Nm
    tau_rate_limit << 30, 30, 30, 15, 15, 15; // Nm/s
    const double max_position_error = 0.10;    // m
    const double max_rotation_error = 0.50;    // rad

    std::signal(SIGINT, requestStop);
    std::signal(SIGTERM, requestStop);
    try {
        pinocchio::Model model;
        pinocchio::urdf::buildModel(urdf_path, model);
        if (model.nq != 6 || model.nv != 6 || !model.existFrame(end_frame)) {
            throw std::runtime_error("URDF 必须是六自由度模型且包含指定末端 frame");
        }
        const auto frame_id = model.getFrameId(end_frame);
        const auto refs = loadReference(input_q, model, frame_id);
        std::vector<LogRow> logs(refs.size());
        std::size_t recorded_count = 0;
        std::filesystem::create_directories(std::filesystem::path(output_txt).parent_path());
        std::ofstream output(output_txt);
        if (!output) throw std::runtime_error("无法创建日志文件：" + output_txt);
        std::cout << "轨迹预计算完成：" << refs.size() << " 点，时长 "
                  << refs.back().time << " s，末端 frame：" << end_frame << '\n';
        if (stop_requested) return 0;

        pinocchio::Data data(model);
        Mat6 J = Mat6::Zero();
        Vec6 previous_tau = Vec6::Zero();
        std::array<double, 6> q{}, dq{}, motor_tau{};
        std::size_t index = 0;
        std::exception_ptr callback_error;
        std::exception_ptr run_error;
        rokae::Torque command(6);
        std::chrono::steady_clock::time_point first_callback;

        rokae::xMateRobot robot;
        robot.connectToRobot(robot_ip, local_ip);
        std::cout << "机器人连接成功\n";
        std::error_code ec;
        robot.setOperateMode(rokae::OperateMode::automatic, ec);
        checkError(ec, "设置自动模式失败");
        // SDK 要求在切换到 RtCommand 之前设置网络容忍阈值。
        robot.setRtNetworkTolerance(20, ec);
        checkError(ec, "设置实时网络容忍阈值失败");
        robot.setMotionControlMode(rokae::MotionControlMode::RtCommand, ec);
        checkError(ec, "设置实时模式失败");
        // 只有进入实时模式后才能创建实时控制器，否则 SDK 会抛出
        // “运动控制模式错误, 非实时模式”。
        auto rt = robot.getRtMotionController().lock();
        if (!rt) throw std::runtime_error("无法获取实时控制器");
        bool loop_started = false;
        bool motion_started = false;
        bool receive_started = false;
        try {
            robot.setPowerState(true, ec);
            checkError(ec, "上电失败");

            std::array<double, 6> q_start{};
            std::copy_n(refs.front().q.data(), 6, q_start.begin());
            const auto q_now = robot.jointPos(ec);
            checkError(ec, "读取初始关节位置失败");
            if (stop_requested) throw std::runtime_error("用户请求停止");
            motion_started = true;
            rt->MoveJ(0.1, q_now, q_start);
            if (stop_requested) throw std::runtime_error("用户请求停止");
            receive_started = true;
            robot.startReceiveRobotState(std::chrono::milliseconds(1),
                {rokae::RtSupportedFields::jointPos_m,
                 rokae::RtSupportedFields::jointVel_m,
                 rokae::RtSupportedFields::motorTau});

            std::function<rokae::Torque(void)> callback = [&]() {
                const auto begin = std::chrono::steady_clock::now();
                if (stop_requested || index >= refs.size()) {
                    command.setFinished();
                    return command;
                }
                try {
                    if (index == 0) first_callback = begin;
                    if (robot.getStateData(rokae::RtSupportedFields::jointPos_m, q) != 0
                        || robot.getStateData(rokae::RtSupportedFields::jointVel_m, dq) != 0
                        || robot.getStateData(rokae::RtSupportedFields::motorTau, motor_tau) != 0) {
                        throw std::runtime_error("实时状态读取失败");
                    }
                    const Vec6 q_actual = Eigen::Map<const Vec6>(q.data());
                    const Vec6 dq_actual = Eigen::Map<const Vec6>(dq.data());
                    if (!q_actual.allFinite() || !dq_actual.allFinite()) {
                        throw std::runtime_error("实际关节状态包含非有限值");
                    }
                    const auto& ref = refs[index];
                    pinocchio::computeJointJacobians(model, data, q_actual);
                    pinocchio::updateFramePlacements(model, data);
                    J.setZero();
                    pinocchio::getFrameJacobian(model, data, frame_id,
                                               pinocchio::LOCAL_WORLD_ALIGNED, J);
                    const auto& actual_pose = data.oMf[frame_id];
                    Vec6 error;
                    error.head<3>() = ref.pose.translation() - actual_pose.translation();
                    // R_actual * log(R_actual^T * R_ref)：在世界轴中表达的最短旋转误差。
                    const Eigen::Matrix3d rotation_error =
                        actual_pose.rotation().transpose() * ref.pose.rotation();
                    error.tail<3>() = actual_pose.rotation() * pinocchio::log3(rotation_error);
                    if (!error.allFinite() || !J.allFinite()
                        || error.head<3>().norm() > max_position_error
                        || error.tail<3>().norm() > max_rotation_error) {
                        throw std::runtime_error("末端跟踪误差超限或运动学数据异常");
                    }
                    const Vec6 velocity_error = ref.velocity - J * dq_actual;
                    const Vec6 wrench = kp.cwiseProduct(error) + kd.cwiseProduct(velocity_error);
                    const Vec6 tau_pd = J.transpose() * wrench;
                    const Vec6 tau_raw = ref.tau_ff + tau_pd;
                    if (!tau_raw.allFinite()) throw std::runtime_error("计算力矩包含非有限值");
                    Vec6 tau_cmd;
                    for (int j = 0; j < 6; ++j) {
                        const double bounded = std::clamp(tau_raw[j], -tau_limit[j], tau_limit[j]);
                        const double delta = tau_rate_limit[j] * sample_dt;
                        tau_cmd[j] = std::clamp(bounded, previous_tau[j] - delta, previous_tau[j] + delta);
                    }
                    previous_tau = tau_cmd;
                    std::copy_n(tau_cmd.data(), 6, command.tau.begin());
                    auto& log = logs[index];
                    log.time = ref.time;
                    log.wall_time = std::chrono::duration<double>(begin - first_callback).count();
                    log.q = q_actual;
                    log.dq = dq_actual;
                    log.pose_error = error;
                    log.tau_ff = ref.tau_ff;
                    log.tau_pd = tau_pd;
                    log.tau_cmd = tau_cmd;
                    log.tau_measured = Eigen::Map<const Vec6>(motor_tau.data());
                    log.limited = (tau_cmd - tau_raw).cwiseAbs().maxCoeff() > 1e-9;
                    log.compute_us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - begin).count();
                    recorded_count = ++index;
                    if (index == refs.size()) command.setFinished();
                } catch (...) {
                    // SDK 通过结束标记停止运动；不从实时回调向外抛异常。
                    callback_error = std::current_exception();
                    command.setFinished();
                }
                return command;
            };
            rt->setControlLoop(callback, 0, true);
            motion_started = true;
            rt->startMove(rokae::RtControllerMode::torque);
            loop_started = true;
            rt->startLoop(true);
        } catch (...) {
            run_error = std::current_exception();
        }
        // 回调引用的所有局部对象仍存活。先停止回调，再保存日志或抛出异常。
        if (loop_started) {
            try { rt->stopLoop(); }
            catch (...) {
                if (!run_error) run_error = std::current_exception();
                rt->disconnectNetwork();
            }
        }
        if (motion_started) {
            try { rt->stopMove(); }
            catch (...) {
                if (!run_error) run_error = std::current_exception();
                rt->disconnectNetwork();
            }
        }
        if (receive_started) robot.stopReceiveRobotState();
        writeLog(output, logs, recorded_count);
        double max_us = 0.0;
        std::size_t limited_count = 0;
        std::size_t overrun_count = 0;
        for (std::size_t i = 0; i < recorded_count; ++i) {
            max_us = std::max(max_us, logs[i].compute_us);
            limited_count += logs[i].limited;
            overrun_count += logs[i].compute_us > 1000.0;
        }
        std::cout << "记录 " << recorded_count << " 点，最大回调计算耗时 " << max_us
                  << " us，超过1ms " << overrun_count << " 次，限幅 " << limited_count
                  << " 次，日志：" << output_txt << '\n';
        if (callback_error) std::rethrow_exception(callback_error);
        if (run_error) std::rethrow_exception(run_error);
        std::cout << (stop_requested ? "用户停止控制\n" : "轨迹控制结束\n");
    } catch (const std::exception& e) {
        std::cerr << "笛卡尔 PD 控制失败：" << e.what() << '\n';
        return 1;
    }
    return 0;
}
