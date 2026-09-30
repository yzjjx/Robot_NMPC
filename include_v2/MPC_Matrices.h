#ifndef MPC_V2_MPC_MATRICES_H
#define MPC_V2_MPC_MATRICES_H

#include "types.h"
#include <vector>

namespace mpc_v2 {

// 当前名义轨迹始终从实测状态积分，所以 delta_x0=0、动力学缺陷=0。
// 因此这里只需 Gamma；本版没有跨周期状态缓存。
class MPCMatrices {
public:
    explicit MPCMatrices(
        int horizon
    );
    void update(
        const std::vector<MatA>& Mat_A,
        const std::vector<MatB>& Mat_B
    );
    Eigen::MatrixXd Gamma;           // Delta_X = Gamma * Delta_U
    Eigen::MatrixXd weighted_Gamma;  // 按行乘以状态权重后的 Gamma
    Eigen::VectorXd state_weights;   // Q_bar 的对角元素，末端使用 F
    Eigen::VectorXd control_weights; // R_bar 的对角元素

private:
    int N;                      // 预测步数
    Eigen::MatrixXd Gamma_i;    // 当前节点的控制影响矩阵
    Eigen::MatrixXd Gamma_next; // 下一节点的递推缓存
};

} // namespace mpc_v2

#endif // MPC_V2_MPC_MATRICES_H
