#pragma once

#include "types.h"
#include <vector>

namespace mpc_v2 {

// 当前名义轨迹始终从实测状态积分，所以 delta_x0=0、动力学缺陷=0。
// 因此这里只需 Gamma；本版没有跨周期状态缓存。
class MPCMatrices {
public:
    explicit MPCMatrices(int horizon);
    void update(const std::vector<MatA>& A, const std::vector<MatB>& B);
    Eigen::MatrixXd Gamma;
    Eigen::MatrixXd weighted_Gamma;
    Eigen::VectorXd state_weights;
    Eigen::VectorXd control_weights;
private:
    int N_;
    Eigen::MatrixXd row_, next_row_;
};

} // namespace mpc_v2
