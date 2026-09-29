#pragma once
#include "mpc_types_v3.h"
#include <qpOASES.hpp>

namespace mpc_v3 {
// 每个反馈相位拥有自己的QP对象和固定H。新模型建立新对象，不原地改旧H。
class PreparedQP {
public:
    using Matrix = Eigen::Matrix<qpOASES::real_t,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
    using Vector = Eigen::Matrix<qpOASES::real_t,Eigen::Dynamic,1>;
    PreparedQP(const Eigen::MatrixXd& H, const Eigen::VectorXd& g,
               const Eigen::MatrixXd& K, const Eigen::VectorXd& lb,
               const Eigen::VectorXd& ub, int max_iterations);
    PreparedQP(const PreparedQP&) = delete;
    PreparedQP& operator=(const PreparedQP&) = delete;
    Eigen::VectorXd solve(const State& dx);
    int iterations() const { return iterations_; }
private:
    Matrix H_; // QProblemB浅引用此存储；必须比solver活得久
    Vector base_g_, g_, lb_, ub_, solution_;
    Eigen::MatrixXd K_;
    qpOASES::QProblemB solver_;
    int max_iterations_, iterations_ = 0;
    bool valid_ = false;
};
} // namespace mpc_v3
