#include "NMPC_control.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>

// 固定测量快照的 CPU 耗时测试，不连接机器人，也不冒充实时闭环。
// 参数：URDF [预测间隔秒=0.1] [预测步数=15] [测量次数=100]
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: mpc_benchmark URDF [dt] [N] [count]\n";
        return 1;
    }
    try {
        const double dt = argc > 2 ? std::stod(argv[2]) : 0.1;
        const int n = mpc_v2::checked_horizon(argc > 3 ? std::stoi(argv[3]) : 15);
        const int count = argc > 4 ? std::stoi(argv[4]) : 100;
        mpc_v2::require(count > 0, "count must be positive");
        mpc_v2::Dynamics dynamics(argv[1]);
        mpc_v2::Controller controller(dynamics, dt, n);
        mpc_v2::State state = mpc_v2::State::Zero();
        state(0) = 0.01;
        Eigen::MatrixXd refs = Eigen::MatrixXd::Zero(n+1, mpc_v2::NX);
        Eigen::MatrixXd ddqs = Eigen::MatrixXd::Zero(n, mpc_v2::DOF);
        // 预热单独处理；首周期耗时在正式控制中仍必须考虑。
        for (int i = 0; i < 5; ++i) controller.compute_control(state, refs, ddqs);
        std::vector<double> totals;
        totals.reserve(count);
        double dynamics_ms = 0, qp_ms = 0, matrix_ms = 0;
        int misses = 0;
        for (int i = 0; i < count; ++i) {
            controller.compute_control(state, refs, ddqs);
            const auto t = controller.timing();
            totals.push_back(t.total_ms);
            dynamics_ms += t.dynamics_ms;
            qp_ms += t.qp_ms;
            matrix_ms += t.matrices_ms;
            if (t.total_ms >= dt*1000) ++misses;
        }
        std::sort(totals.begin(), totals.end());
        const double mean = std::accumulate(totals.begin(), totals.end(), 0.0)/count;
        const int p99 = std::max(0, static_cast<int>(std::ceil(0.99*count))-1);
        std::cout << std::fixed << std::setprecision(3)
                  << "Configured period ms: " << 1000*dt << "\n"
                  << "Mean/P99/max compute ms: " << mean << " / " << totals[p99]
                  << " / " << totals.back() << "\n"
                  << "Mean dynamics/matrices/QP ms: " << dynamics_ms/count << " / "
                  << matrix_ms/count << " / " << qp_ms/count << "\n"
                  << "Compute-budget misses: " << misses << "/" << count << "\n"
                  << "This is NOT a real-robot control-frequency measurement.\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
