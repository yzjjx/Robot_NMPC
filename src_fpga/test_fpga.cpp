#include "fpga_backend.h"
#include "mpc_controller_v3.h"
#include <iostream>

using namespace mpc_fpga;
namespace {
void check(bool ok,const char* text) {if(!ok) throw std::runtime_error(text);}
template<class F> void rejected(F call) {bool fail=false;try{call();}catch(const std::exception&){fail=true;}check(fail,"Expected rejection");}
void samples(Eigen::MatrixXd& X,Eigen::MatrixXd& U) {
    X.setZero(37,NX);U.setZero(37,NU);
    for(int j=0;j<18;++j) {
        if(j<NX) {X(1+2*j,j)=0.02;X(2+2*j,j)=-0.02;}
        else {U(1+2*j,j-NX)=0.1;U(2+2*j,j-NX)=-0.1;}
    }
}
void codec_test() {
    check(quantize(0.5,0)==0 && quantize(1.5,0)==2 && quantize(-1.5,0)==-2,"Round-to-even failed");
    State x=State::Zero();Joint u=Joint::Zero();x(0)=1;x(6)=-1;u(0)=1;
    const auto bytes=pack_inputs({x},{u});
    check(bytes.size()==72 && bytes[3]==0x10 && bytes[27]==0xf0 && bytes[51]==0x01,"Input layout failed");
    std::vector<std::uint8_t> output(24,0);output[2]=1;
    check(std::abs(unpack_outputs(output)[0](0)-1)<1e-12,"Q16.16 decode failed");
    output[0]=0xff;output[1]=0xff;output[2]=0xff;output[3]=0x7f;
    rejected([&]{unpack_outputs(output);});
    u(0)=11;rejected([&]{pack_inputs({x},{u});});
    rejected([&]{pack_inputs({},{});});
}
// 检查模型不一致时确实拒绝，而不是只打印警告后继续。
class BadDevice final : public AbaDevice {
public:
    std::vector<Joint> run(const std::vector<State>& x,const std::vector<Joint>&) override {return std::vector<Joint>(x.size(),Joint::Constant(10000));}
    BackendStats stats() const override {return {};}
    void clear_stats() override {}
    std::string name() const override {return "bad_test_device";}
};
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"Usage: test_fpga_mock URDF");codec_test();
        Config c;c.horizon=3;c.enforce_feedback_budget=false;
        FpgaBackend backend(argv[1],c,std::make_unique<MockDevice>(argv[1],c));
        State x=State::Zero();Joint u=Joint::Zero();
        rejected([&]{backend.integrate_nodes({x},{u},{0.01});});
        Eigen::MatrixXd X,U;samples(X,U);backend.validate_model(X,U,0.05,0.002);
        CpuBackend cpu(argv[1],c);
        x(0)=0.1;u(0)=0.01;
        const auto fpga=backend.integrate_nodes({x,x},{u,u},{0.01,0.003333333333333333});
        const auto reference=cpu.integrate_nodes({x,x},{u,u},{0.01,0.003333333333333333});
        for(int i=0;i<2;++i) {
            check((fpga[i].next-reference[i].next).norm()<1e-4,"Mock forward integration mismatch");
            check((fpga[i].A-reference[i].A).norm()/(1+reference[i].A.norm())<1e-3,"Mock sensitivity mismatch");
        }
        check(backend.stats().samples>0 && backend.stats().batches>1,"Batched backend not used");
        FpgaBackend bad(argv[1],c,std::make_unique<BadDevice>());
        rejected([&]{bad.validate_model(X,U,0.05,0.002);});check(!bad.ready(),"Mismatch did not invalidate backend");
        // 真实控制器也必须经过后端检查，随后与CPU使用同一个反馈实现。
        auto device=std::make_unique<FpgaBackend>(argv[1],c,std::make_unique<MockDevice>(argv[1],c));
        Controller controller(c,std::move(device));controller.validate_backend(X,U);
        Eigen::MatrixXd refs=Eigen::MatrixXd::Zero(4,NX),ddq=Eigen::MatrixXd::Zero(3,NU);
        controller.prepare(0,State::Zero(),refs,ddq);
        check(controller.feedback(State::Zero(),0).torque.allFinite(),"Mock controller failed");
        std::cout<<"Codec and mock checks passed; no physical FPGA was accessed\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
