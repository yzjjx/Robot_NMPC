#include "fpga_backend.h"
#include <algorithm>

namespace mpc_fpga {
FpgaBackend::FpgaBackend(const std::string& urdf,const Config& c,std::unique_ptr<AbaDevice> device)
    :derivatives_(urdf,c.gravity_compensated,c.integration_step),step_(c.integration_step),device_(std::move(device)) {
    c.validate(); require(device_!=nullptr,"Missing ABA device");
    require(c.torque_limit<=10.0,"Current HLS deployment domain requires torque_limit <= 10 Nm");
}
Joint FpgaBackend::reference_torque(const State& x,const Joint& ddq) {
    return derivatives_.inverse_dynamics(x,ddq);
}
std::vector<Joint> FpgaBackend::evaluate(const std::vector<State>& x,const std::vector<Joint>& u) {
    require(x.size()==u.size(),"ABA batch shape mismatch");
    std::vector<Joint> all; all.reserve(x.size());
    for(std::size_t begin=0;begin<x.size();begin+=MAX_BATCH) {
        const auto end=std::min(begin+MAX_BATCH,x.size());
        auto result=device_->run(std::vector<State>(x.begin()+begin,x.begin()+end),
                                 std::vector<Joint>(u.begin()+begin,u.begin()+end));
        require(result.size()==end-begin,"ABA device result count mismatch");
        for(const auto& ddq:result) if(!ddq.allFinite()) throw std::runtime_error("Nonfinite device acceleration");
        all.insert(all.end(),result.begin(),result.end());
    }
    return all;
}
void FpgaBackend::validate_model(const Eigen::MatrixXd& X,const Eigen::MatrixXd& U,double atol,double rtol) {
    validated_=false;
    require(X.rows()>=19 && X.cols()==NX && U.rows()==X.rows() && U.cols()==NU &&
            X.allFinite() && U.allFinite(),"Preflight needs at least 19 representative state/torque samples");
    require(std::isfinite(atol) && std::isfinite(rtol) && atol>0 && rtol>=0,"Invalid preflight tolerances");
    // 避免全零重复样本碰巧通过。仍须由用户覆盖实际运动区域。
    require(((X.colwise().maxCoeff()-X.colwise().minCoeff()).array()>1e-5).all() &&
            ((U.colwise().maxCoeff()-U.colwise().minCoeff()).array()>1e-5).all(),
            "Preflight must vary every q, dq and torque dimension");
    std::vector<State> x; std::vector<Joint> u;
    for(int i=0;i<X.rows();++i) { x.push_back(X.row(i).transpose());u.push_back(U.row(i).transpose()); }
    const auto values=evaluate(x,u);
    double worst=0;
    for(std::size_t i=0;i<x.size();++i) {
        // 对量化后的相同输入比较，避免把输入舍入差误判成模型差。
        const auto quantized=quantized_input(x[i],u[i]);
        const Joint expected=derivatives_.acceleration(quantized.first,quantized.second);
        require(expected.allFinite(), "Nonfinite Pinocchio reference during preflight");
        for(int j=0;j<NU;++j)
            worst=std::max(worst,std::abs(values[i](j)-expected(j))/(atol+rtol*std::abs(expected(j))));
    }
    if(worst>1) throw std::runtime_error("FPGA/Pinocchio model mismatch, normalized error="+std::to_string(worst)+
        ". Check gravity, inertias, joint convention, bit version and Q formats; do not bypass this check.");
    validated_=true; device_->clear_stats();
}

std::vector<Linearization> FpgaBackend::integrate_nodes(const std::vector<State>& x,
    const std::vector<Joint>& u,const std::vector<double>& durations) {
    if(!validated_) throw std::runtime_error("Run representative FPGA model validation first");
    require(x.size()==u.size() && x.size()==durations.size(),"Integration batch shape mismatch");
    std::vector<Linearization> result; std::vector<int> counts; std::vector<double> h;
    int max_steps=0;
    for(std::size_t i=0;i<x.size();++i) {
        check_domain(x[i],u[i]);
        const double count=std::ceil(durations[i]/step_);
        require(std::isfinite(durations[i]) && durations[i]>=0 && count<=1000000,"Invalid integration duration");
        const int steps=durations[i]==0?0:std::max(1,static_cast<int>(count));
        counts.push_back(steps); h.push_back(steps?durations[i]/steps:0);
        result.push_back({x[i],MatA::Identity(),MatB::Zero()}); max_steps=std::max(max_steps,steps);
    }
    try {
        for(int step=0;step<max_steps;++step) {
            std::vector<State> ready_x; std::vector<Joint> ready_u; std::vector<std::size_t> ids;
            for(std::size_t i=0;i<x.size();++i) if(step<counts[i]) {
                ready_x.push_back(result[i].next); ready_u.push_back(u[i]); ids.push_back(i);
            }
            const auto accelerations=evaluate(ready_x,ready_u);
            for(std::size_t j=0;j<ids.size();++j) {
                const auto i=ids[j]; const State current=result[i].next;
                // 复用v2解析敏感度实现。这个CPU调用仍有动力学计算成本，不承诺FPGA一定更快。
                const auto sensitivity=derivatives_.integrate_linearized(current,u[i],h[i]);
                result[i].A=(sensitivity.A*result[i].A).eval();
                result[i].B=(sensitivity.A*result[i].B+sensitivity.B).eval();
                result[i].next.head<NU>()=current.head<NU>()+h[i]*current.tail<NU>()+0.5*h[i]*h[i]*accelerations[j];
                result[i].next.tail<NU>()=current.tail<NU>()+h[i]*accelerations[j];
                if(!result[i].next.allFinite() || !result[i].A.allFinite() || !result[i].B.allFinite())
                    throw std::runtime_error("Nonfinite FPGA integrated model");
                check_domain(result[i].next,u[i]);
            }
        }
    } catch(...) { validated_=false; throw; }
    return result;
}
} // namespace mpc_fpga
