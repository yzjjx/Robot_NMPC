#include "python_api_v3.h"
#include "fpga_backend.h"
PYBIND11_MODULE(rokae_mpc_fpga,m) {
    m.doc()="Same preparation/feedback MPC with current XDMA ABA; hardware access is opt-in";
    mpc_v3::bind_api(m,[](const std::string& urdf,const mpc_v3::Config& c,const std::string& backend,
                         bool allow,const std::string& h2c,const std::string& c2h)->std::unique_ptr<mpc_v3::DynamicsBackend> {
        if(backend=="cpu") return std::make_unique<mpc_v3::CpuBackend>(urdf,c);
        std::unique_ptr<mpc_fpga::AbaDevice> device;
        if(backend=="mock") device=std::make_unique<mpc_fpga::MockDevice>(urdf,c);
        else if(backend=="fpga") {
            mpc_v3::require(allow,"Actual device access requires allow_hardware=True");
            mpc_v3::require(c.torque_limit<=10,"Use the current FPGA deployment torque domain <=10 Nm");
            mpc_fpga::DeviceOptions options; options.h2c=h2c;options.c2h=c2h;
            device=std::make_unique<mpc_fpga::XdmaDevice>(options);
        } else throw std::invalid_argument("backend must be cpu, mock or fpga");
        return std::make_unique<mpc_fpga::FpgaBackend>(urdf,c,std::move(device));
    },"mock"); // 默认绝不打开板卡
}
