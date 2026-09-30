#ifndef MPC_V2_TYPES_H
#define MPC_V2_TYPES_H

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

namespace mpc_v2 {

// 本版只针对原工程的 6 个单自由度转动关节。
constexpr int DOF = 6;
constexpr int NX = 12;
using Joint = Eigen::Matrix<double, DOF, 1>;
using State = Eigen::Matrix<double, NX, 1>;
using MatA = Eigen::Matrix<double, NX, NX>;
using MatB = Eigen::Matrix<double, NX, DOF>;

inline void require(
    bool condition,
    const char* message
)
{
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

// 构造函数分配内存前先检查参数。上界只是避免误输入造成超大分配。
inline int checked_horizon(
    int prediction_steps
)
{
    require(prediction_steps >= 1 && prediction_steps <= 500, "horizon must be in [1, 500]");
    return prediction_steps;
}

struct Timing {
    double reference_ms = 0;
    double dynamics_ms = 0; // 名义轨迹 + 解析线性化
    double matrices_ms = 0; // Gamma + QP 代价和上下界
    double qp_ms = 0;
    double total_ms = 0; // C++ compute_control 的墙钟时间
    int qp_iterations = 0;
    bool qp_success = false;
};

} // namespace mpc_v2

#endif // MPC_V2_TYPES_H
