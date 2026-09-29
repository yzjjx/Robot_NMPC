#include "dynamics_backend.h"
namespace mpc_v3 {
CpuBackend::CpuBackend(const std::string& urdf, const Config& config)
    : dynamics_(urdf,config.gravity_compensated,config.integration_step) {}

std::vector<Linearization> CpuBackend::integrate_nodes(const std::vector<State>& x,
    const std::vector<Joint>& u, const std::vector<double>& durations) {
    require(x.size()==u.size() && x.size()==durations.size(), "Batch shape mismatch");
    std::vector<Linearization> result;
    result.reserve(x.size());
    for (std::size_t i=0; i<x.size(); ++i)
        result.push_back(dynamics_.integrate_linearized(x[i],u[i],durations[i]));
    return result;
}
Joint CpuBackend::reference_torque(const State& x, const Joint& ddq) {
    return dynamics_.inverse_dynamics(x,ddq);
}
} // namespace mpc_v3
