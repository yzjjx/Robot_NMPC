#include "mpc_controller_v3.h"
#include <iostream>

using namespace mpc_v3;
namespace {
void check(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
template<class F> void rejected(F call) {
    bool fail=false; try {call();} catch(const std::exception&) {fail=true;}
    check(fail,"Expected rejection");
}
// 独立可手算的双积分器，用于验证时间、缺陷和反馈，不依赖机器人参数。
class LinearBackend final : public DynamicsBackend {
public:
    int calls=0;
    std::vector<Linearization> integrate_nodes(const std::vector<State>& x,
        const std::vector<Joint>& u,const std::vector<double>& dt) override {
        ++calls; std::vector<Linearization> values;
        for(std::size_t i=0;i<x.size();++i) {
            MatA a=MatA::Identity();MatB b=MatB::Zero();
            a.topRightCorner<NU,NU>()=dt[i]*Eigen::Matrix<double,NU,NU>::Identity();
            b.topRows<NU>()=0.5*dt[i]*dt[i]*Eigen::Matrix<double,NU,NU>::Identity();
            b.bottomRows<NU>()=dt[i]*Eigen::Matrix<double,NU,NU>::Identity();
            values.push_back({a*x[i]+b*u[i],a,b});
        }
        return values;
    }
    Joint reference_torque(const State&,const Joint& ddq) override {return ddq;}
    void validate_model(const Eigen::MatrixXd&,const Eigen::MatrixXd&,double,double) override {}
    bool ready() const override {return true;}
    std::string name() const override {return "test_double_integrator";}
};
void condensing_test() {
    std::vector<MatA> a(3,MatA::Identity()); std::vector<MatB> b(3,MatB::Zero());
    std::vector<State> d(3,State::Zero());
    for(int i=0;i<3;++i) {
        a[i](0,6)=0.01; b[i](6,0)=0.01; d[i](0)=0.02*(i+1);
    }
    Eigen::MatrixXd phi,gamma;Eigen::VectorXd eta;
    condense(a,b,d,phi,gamma,eta);
    State dx=State::Constant(0.001),current=dx;
    const Eigen::VectorXd du=Eigen::VectorXd::LinSpaced(18,-0.1,0.1);
    const Eigen::VectorXd total=phi*dx+gamma*du+eta;
    for(int i=0;i<3;++i) {
        current=(a[i]*current+b[i]*du.segment<NU>(i*NU)+d[i]).eval();
        check((current-total.segment<NX>(i*NX)).norm()<1e-12,"Affine condensing failed");
    }
}
void phases_test() {
    Config c;c.horizon=4;c.enforce_feedback_budget=false;
    LinearBackend backend;
    Eigen::MatrixXd x=Eigen::MatrixXd::Zero(5,NX),u=Eigen::MatrixXd::Zero(4,NU);
    x(0,0)=0.01;x(0,6)=0.02;
    auto block=prepare_block(backend,c,0,x,u,Eigen::MatrixXd::Zero(5,NX),u);
    for(int p=0;p<3;++p) {
        auto& model=*block->phases[p];
        check(std::abs(model.first_duration-(c.model_dt-p*c.feedback_dt()))<1e-12,"Wrong phase duration");
        State dx=State::Zero();dx(0)=0.001;
        const auto du=model.qp->solve(dx);
        const Eigen::VectorXd correction=model.Phi*dx+model.Gamma*du+model.eta;
        State actual=model.x0+dx;
        for(int i=0;i<c.horizon;++i) {
            const double h=i==0?model.first_duration:c.model_dt;
            const Joint tau=du.segment<NU>(i*NU);
            actual.head<NU>()+=h*actual.tail<NU>()+0.5*h*h*tau;
            actual.tail<NU>()+=h*tau;
            const State prediction=model.nominal_x.row(i+1).transpose()+correction.segment<NX>(i*NX);
            check((actual-prediction).norm()<1e-9,"Phase prediction/time or defect is wrong");
        }
    }
}
void controller_test() {
    Config c;c.horizon=4;c.enforce_feedback_budget=false;
    auto backend=std::make_unique<LinearBackend>();auto* counter=backend.get();
    Controller mpc(c,std::move(backend));
    Eigen::MatrixXd refs=Eigen::MatrixXd::Zero(5,NX),ddq=Eigen::MatrixXd::Zero(4,NU);
    State x=State::Zero();x(0)=0.01;
    rejected([&]{mpc.feedback(x,0);});
    mpc.prepare(0,x,refs,ddq);
    const int before=counter->calls;
    const auto a=mpc.feedback(x,0);
    check(counter->calls==before,"Feedback called dynamics");
    x(0)+=0.001;
    const auto b=mpc.feedback(x,1);
    check((a.torque-b.torque).norm()>1e-8,"New measurement did not affect feedback");
    check(std::abs(b.state_times(1)-0.01)<1e-12,"Future time grid moved by wrong amount");
    rejected([&]{mpc.feedback(x,1);});
    rejected([&]{mpc.feedback(x,2,0.1);});
    State far=x;far(0)+=1;rejected([&]{mpc.feedback(far,2);});
    const auto last=mpc.feedback(x,2);
    rejected([&]{mpc.feedback(x,3);}); // 不自动用上一块的旧模型
    mpc.prepare(1,last.states.row(1).transpose(),refs,ddq);
    mpc.feedback(last.states.row(1).transpose(),3);
    rejected([&]{mpc.prepare(0,x,refs,ddq);});
    mpc.reset();check(!mpc.has_model(0) && !mpc.has_model(1),"Reset failed");
}
}
int main() {
    try {condensing_test();phases_test();controller_test();std::cout<<"CPU mathematical/scheduling checks passed\n";}
    catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
