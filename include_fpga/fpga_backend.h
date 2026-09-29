#pragma once
#include "aba_device.h"
#include "../include_v3/dynamics_backend.h"

namespace mpc_fpga {
// FPGA给出加速度；CPU积分并计算平滑Pinocchio模型的解析导数。
// 它是经数值一致性检查的近似雅可比，不是定点量化函数的严格导数。
class FpgaBackend final : public DynamicsBackend {
public:
    FpgaBackend(const std::string& urdf,const Config& config,std::unique_ptr<AbaDevice> device);
    std::vector<Linearization> integrate_nodes(const std::vector<State>& x,
        const std::vector<Joint>& u,const std::vector<double>& durations) override;
    Joint reference_torque(const State& x,const Joint& ddq) override;
    void validate_model(const Eigen::MatrixXd& states,const Eigen::MatrixXd& torques,
                        double absolute_tolerance,double relative_tolerance) override;
    bool ready() const override { return validated_; }
    std::string name() const override { return device_->name(); }
    BackendStats stats() const override { return device_->stats(); }
    void clear_stats() override { device_->clear_stats(); }
private:
    mpc_v2::Dynamics derivatives_;
    double step_;
    std::unique_ptr<AbaDevice> device_;
    bool validated_=false;
    std::vector<Joint> evaluate(const std::vector<State>& x,const std::vector<Joint>& u);
};
} // namespace mpc_fpga
