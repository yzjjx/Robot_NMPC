#ifndef MPC_V2_PINOCCHIO_FUN_H
#define MPC_V2_PINOCCHIO_FUN_H

#include "types.h"
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <string>

namespace mpc_v2 {

struct Linearization {
    State next; // 保持当前力矩积分后的下一状态
    MatA A;     // 整个积分区间对初始状态的导数
    MatB B;     // 整个积分区间对保持力矩的导数
};

// 一个 Dynamics 对象含有可变的 Pinocchio Data，不可跨线程同时调用。
class Dynamics {
public:
    explicit Dynamics(
        const std::string& urdf_path,
        bool gravity_compensated = true,
        double integration_step = 0.001
    );

    // ABA：输入状态和附加力矩，输出关节加速度。
    Joint acceleration(
        const State& state,
        const Joint& control
    );
    // RNEA：输入状态和期望加速度，输出对应的附加力矩。
    Joint inverse_dynamics(
        const State& state,
        const Joint& ddq
    );
    // 对应旧版 compute_held_state：保持力矩不变，分小步积分。
    State integrate(
        const State& state,
        const Joint& control,
        double duration
    );
    // 同时生成下一状态和离散 A/B，使用解析导数代替旧版前向差分。
    Linearization integrate_linearized(
        const State& state,
        const Joint& control,
        double duration
    );

    double integration_step() const
    {
        return max_integration_step;
    }

private:
    pinocchio::Model model;      // 机器人模型
    pinocchio::Data data;        // 动力学递推缓存
    double max_integration_step; // 积分小步长的上限
    int step_count(
        double duration
    ) const;
    static State advance(
        const State& state,
        const Joint& ddq,
        double integration_dt
    );
};

} // namespace mpc_v2

#endif // MPC_V2_PINOCCHIO_FUN_H
