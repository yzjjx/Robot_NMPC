#include "prepared_model.h"
#include <algorithm>

namespace mpc_v3 {
void condense(const std::vector<MatA>& A, const std::vector<MatB>& B,
    const std::vector<State>& d, Eigen::MatrixXd& Phi, Eigen::MatrixXd& Gamma, Eigen::VectorXd& eta) {
    const int N=static_cast<int>(A.size());
    require(N>0 && B.size()==A.size() && d.size()==A.size(), "Condensing shape mismatch");
    Phi.resize(N*NX,NX); Gamma.setZero(N*NX,N*NU); eta.resize(N*NX);
    MatA p=MatA::Identity();
    Eigen::MatrixXd g=Eigen::MatrixXd::Zero(NX,N*NU);
    State e=State::Zero();
    for(int i=0;i<N;++i) {
        p=(A[i]*p).eval();
        g=(A[i]*g).eval();
        g.block(0,i*NU,NX,NU)+=B[i];
        e=(A[i]*e+d[i]).eval();
        Phi.middleRows(i*NX,NX)=p;
        Gamma.middleRows(i*NX,NX)=g;
        eta.segment<NX>(i*NX)=e;
    }
}

std::shared_ptr<PreparedBlock> prepare_block(DynamicsBackend& backend,const Config& c,
    std::int64_t block,const Eigen::MatrixXd& X,const Eigen::MatrixXd& U,
    const Eigen::MatrixXd& Xref,const Eigen::MatrixXd& Uref) {
    const auto started=Clock::now();
    c.validate();
    const int N=c.horizon;
    require(X.rows()==N+1 && X.cols()==NX && U.rows()==N && U.cols()==NU &&
            Xref.rows()==N+1 && Xref.cols()==NX && Uref.rows()==N && Uref.cols()==NU,
            "Preparation shape mismatch");
    require(X.allFinite() && U.allFinite() && Xref.allFinite() && Uref.allFinite(),
            "Nonfinite preparation data");
    auto out=std::make_shared<PreparedBlock>(); out->index=block; out->info.block=block;
    std::vector<State> x(N); std::vector<Joint> u(N);
    for(int i=0;i<N;++i) { x[i]=X.row(i).transpose(); u[i]=U.row(i).transpose(); }
    backend.clear_stats();
    // 多重射击：各区间的起点已知，可批量计算，不串行滚动整个时域。
    const auto base=backend.integrate_nodes(x,u,std::vector<double>(N,c.model_dt));
    require(base.size()==x.size(), "Backend returned wrong interval count");
    std::vector<State> phase_starts{x[0]};
    std::vector<Linearization> first_maps{base[0]};
    if(c.feedback_ratio>1) {
        const int P=c.feedback_ratio-1;
        std::vector<double> elapsed(P),remaining(P);
        for(int p=1;p<=P;++p) { elapsed[p-1]=p*c.feedback_dt(); remaining[p-1]=c.model_dt-elapsed[p-1]; }
        const auto starts=backend.integrate_nodes(std::vector<State>(P,x[0]),std::vector<Joint>(P,u[0]),elapsed);
        require(starts.size()==static_cast<std::size_t>(P), "Backend returned wrong phase count");
        std::vector<State> sx;
        for(const auto& s:starts) { phase_starts.push_back(s.next); sx.push_back(s.next); }
        const auto maps=backend.integrate_nodes(sx,std::vector<Joint>(P,u[0]),remaining);
        require(maps.size()==sx.size(), "Backend returned wrong first-interval count");
        first_maps.insert(first_maps.end(),maps.begin(),maps.end());
    }
    out->info.model_ms=elapsed_ms(started);
    const auto cost_start=Clock::now();
    for(int p=0;p<c.feedback_ratio;++p) {
        auto phase=std::make_shared<PhaseModel>();
        phase->x0=phase_starts[p]; phase->nominal_x=X; phase->nominal_u=U;
        phase->nominal_x.row(0)=phase->x0.transpose();
        phase->first_duration=c.model_dt-p*c.feedback_dt();
        std::vector<MatA> A(N); std::vector<MatB> B(N); std::vector<State> defects(N);
        Eigen::VectorXd q(N*NX),r=Eigen::VectorXd::Constant(N*NU,0.01);
        Eigen::VectorXd error(N*NX),u_error(N*NU),lb(N*NU),ub(N*NU);
        for(int i=0;i<N;++i) {
            const auto& f=(i==0)?first_maps[p]:base[i];
            require(f.next.allFinite() && f.A.allFinite() && f.B.allFinite(), "Nonfinite backend model");
            A[i]=f.A; B[i]=f.B;
            defects[i]=f.next-X.row(i+1).transpose();
            out->info.max_position_defect=std::max(out->info.max_position_defect,defects[i].head<NU>().cwiseAbs().maxCoeff());
            out->info.max_velocity_defect=std::max(out->info.max_velocity_defect,defects[i].tail<NU>().cwiseAbs().maxCoeff());
            const double terminal=(i==N-1)?2.0:1.0;
            q.segment<NU>(i*NX).setConstant(100*terminal);
            q.segment<NU>(i*NX+NU).setConstant(terminal);
            error.segment<NX>(i*NX)=X.row(i+1).transpose()-Xref.row(i+1).transpose();
            u_error.segment<NU>(i*NU)=U.row(i).transpose()-Uref.row(i).transpose();
            for(int j=0;j<NU;++j) {
                lb(i*NU+j)=std::max(-c.torque_limit-U(i,j),-c.torque_trust);
                ub(i*NU+j)=std::min( c.torque_limit-U(i,j), c.torque_trust);
            }
        }
        if(out->info.max_position_defect>c.max_position_defect || out->info.max_velocity_defect>c.max_velocity_defect)
            throw std::runtime_error("Nominal trajectory defect too large; rebuild a closer nominal trajectory");
        condense(A,B,defects,phase->Phi,phase->Gamma,phase->eta);
        // delta_X = Phi*delta_x0 + Gamma*delta_U + eta。
        const Eigen::MatrixXd GQ=phase->Gamma.transpose()*q.asDiagonal();
        Eigen::MatrixXd H=2.0*GQ*phase->Gamma;
        H.diagonal().array()+=2.0*r.array()+1e-6;
        const Eigen::VectorXd g=2.0*(GQ*(error+phase->eta)+r.cwiseProduct(u_error));
        const Eigen::MatrixXd K=2.0*GQ*phase->Phi;
        phase->qp=std::make_unique<PreparedQP>(H,g,K,lb,ub,c.qp_iterations);
        out->phases.push_back(std::move(phase));
    }
    out->info.matrices_and_qp_ms=elapsed_ms(cost_start);
    out->info.total_ms=elapsed_ms(started);
    out->info.backend=backend.stats();
    return out;
}
} // namespace mpc_v3
