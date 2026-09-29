#pragma once
#include "mpc_types_v3.h"
#include <memory>

namespace mpc_v3 {
// 所有后端都只由准备线程访问。反馈线程绝不调用动力学/PCIe。
class DynamicsBackend {
public:
    virtual ~DynamicsBackend() = default;
    virtual std::vector<Linearization> integrate_nodes(
        const std::vector<State>& x, const std::vector<Joint>& u,
        const std::vector<double>& durations) = 0;
    virtual Joint reference_torque(const State& x, const Joint& ddq) = 0;
    virtual void validate_model(const Eigen::MatrixXd& states, const Eigen::MatrixXd& torques,
                                double absolute_tolerance, double relative_tolerance) = 0;
    virtual bool ready() const = 0;
    virtual std::string name() const = 0;
    virtual BackendStats stats() const { return {}; }
    virtual void clear_stats() {}
};

class CpuBackend final : public DynamicsBackend {
public:
    CpuBackend(const std::string& urdf, const Config& config);
    std::vector<Linearization> integrate_nodes(const std::vector<State>& x,
        const std::vector<Joint>& u, const std::vector<double>& durations) override;
    Joint reference_torque(const State& x, const Joint& ddq) override;
    void validate_model(const Eigen::MatrixXd&, const Eigen::MatrixXd&, double, double) override {}
    bool ready() const override { return true; }
    std::string name() const override { return "pinocchio_cpu"; }
private:
    mpc_v2::Dynamics dynamics_;
};
} // namespace mpc_v3
