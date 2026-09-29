#include "python_api_v3.h"
PYBIND11_MODULE(rokae_mpc_v3,m) {
    m.doc()="CPU preparation/feedback MPC; no robot I/O";
    mpc_v3::bind_api(m,[](const std::string& urdf,const mpc_v3::Config& c,const std::string& backend,
                         bool,const std::string&,const std::string&)->std::unique_ptr<mpc_v3::DynamicsBackend> {
        mpc_v3::require(backend=="cpu","CPU module supports only backend='cpu'");
        return std::make_unique<mpc_v3::CpuBackend>(urdf,c);
    },"cpu");
}
