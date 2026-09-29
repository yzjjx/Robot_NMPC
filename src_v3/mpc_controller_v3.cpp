#include "mpc_controller_v3.h"
#include <algorithm>

namespace mpc_v3 {
Controller::Controller(Config config,std::unique_ptr<DynamicsBackend> backend)
    :config_(config),backend_(std::move(backend)),backend_name_(backend_?backend_->name():"") {
    config_.validate(); require(backend_!=nullptr,"Missing dynamics backend");
}

PreparationInfo Controller::prepare(std::int64_t block,const State& anchor,
    const Eigen::MatrixXd& refs,const Eigen::MatrixXd& ddq_ref) {
    const auto started=Clock::now();
    std::lock_guard<std::mutex> prepare_lock(prepare_mutex_);
    require(block>=0 && block<1000000000 && anchor.allFinite(),"Invalid block or anchor");
    const int N=config_.horizon;
    require(refs.rows()==N+1 && refs.cols()==NX && ddq_ref.rows()==N && ddq_ref.cols()==NU &&
            refs.allFinite() && ddq_ref.allFinite(),"Invalid reference dimensions/values");
    if(!backend_->ready()) throw std::runtime_error("Backend not validated/ready");
    std::shared_ptr<const FeedbackResult> previous;
    {
        std::lock_guard<std::mutex> lock(model_mutex_);
        previous=last_result_;
        if(last_tick_>=0 && block<last_tick_/config_.feedback_ratio)
            throw std::runtime_error("Refusing an obsolete preparation request");
    }
    Eigen::MatrixXd X=refs,U(N,NU),Uref(N,NU);
    for(int i=0;i<N;++i) {
        const State xref=refs.row(i).transpose();
        const Joint ddq=ddq_ref.row(i).transpose();
        Uref.row(i)=backend_->reference_torque(xref,ddq).transpose();
    }
    U=Uref;
    if(previous && previous->block+1==block) {
        // 每跨过一个完整模型周期才移位一次，绝不是每3.33ms移位10ms。
        for(int i=1;i<N;++i) X.row(i)=previous->states.row(i+1);
        for(int i=0;i<N;++i) U.row(i)=previous->controls.row(std::min(i+1,N-1));
        // 最后一个新状态节点用参考补齐；不满足动力学的部分由缺陷项处理。
    }
    X.row(0)=anchor.transpose();
    U=U.cwiseMax(-config_.torque_limit).cwiseMin(config_.torque_limit);
    auto model=prepare_block(*backend_,config_,block,X,U,refs,Uref);
    model->info.total_ms=elapsed_ms(started);
    const auto info=model->info;
    {
        std::lock_guard<std::mutex> lock(model_mutex_);
        if(last_tick_>=0 && block<last_tick_/config_.feedback_ratio)
            throw std::runtime_error("Preparation finished too late; result discarded");
        models_[block]=std::move(model); // 完整模型一次性发布，反馈不会读到半成品
        while(models_.size()>3) models_.erase(models_.begin());
    }
    return info;
}

FeedbackResult Controller::feedback(const State& measured,std::int64_t tick,double age) {
    const auto started=Clock::now();
    std::lock_guard<std::mutex> feedback_lock(feedback_mutex_);
    require(tick>=0 && tick<1000000000 && measured.allFinite(),"Invalid tick or measurement");
    require(std::isfinite(age) && age>=0 && age<=config_.max_measurement_age,"Stale measurement");
    const auto block=tick/config_.feedback_ratio;
    const int phase_index=static_cast<int>(tick%config_.feedback_ratio);
    std::shared_ptr<PreparedBlock> model;
    {
        std::lock_guard<std::mutex> lock(model_mutex_);
        require(tick>last_tick_,"Feedback ticks must increase; reset for a new experiment");
        const auto it=models_.find(block);
        if(it==models_.end()) throw std::runtime_error("Prepared model missing; do not apply an old result");
        model=it->second;
    }
    auto& p=*model->phases[phase_index];
    const State dx=measured-p.x0;
    if(dx.head<NU>().cwiseAbs().maxCoeff()>config_.position_trust ||
       dx.tail<NU>().cwiseAbs().maxCoeff()>config_.velocity_trust)
        throw std::runtime_error("Measured state outside model trust region; refresh or use validated fallback");
    const auto qp_start=Clock::now();
    const Eigen::VectorXd du=p.qp->solve(dx);
    FeedbackResult result;
    result.solve_ms=elapsed_ms(qp_start); result.qp_iterations=p.qp->iterations();
    result.tick=tick; result.block=block; result.phase=phase_index;
    result.states=p.nominal_x; result.controls=p.nominal_u;
    const Eigen::VectorXd change=p.Phi*dx+p.Gamma*du+p.eta;
    result.states.row(0)=measured.transpose();
    result.state_times.resize(config_.horizon+1);
    result.state_times(0)=tick*config_.feedback_dt();
    for(int i=0;i<config_.horizon;++i) {
        result.states.row(i+1)+=change.segment<NX>(i*NX).transpose();
        result.controls.row(i)+=du.segment<NU>(i*NU).transpose();
        result.state_times(i+1)=(block+i+1)*config_.model_dt;
    }
    if(!result.states.allFinite() || !result.controls.allFinite() ||
       result.controls.cwiseAbs().maxCoeff()>config_.torque_limit+1e-5)
        throw std::runtime_error("Invalid predicted state/control sequence");
    result.torque=result.controls.row(0).transpose();
    result.total_ms=elapsed_ms(started);
    result.deadline_missed=result.total_ms>1000*config_.feedback_dt();
    if(config_.enforce_feedback_budget && result.deadline_missed)
        throw std::runtime_error("Feedback compute budget exceeded; result rejected");
    {
        std::lock_guard<std::mutex> lock(model_mutex_);
        last_tick_=tick;
        last_result_=std::make_shared<FeedbackResult>(result);
    }
    return result;
}

void Controller::validate_backend(const Eigen::MatrixXd& states,const Eigen::MatrixXd& torques,
                                 double atol,double rtol) {
    std::lock_guard<std::mutex> lock(prepare_mutex_);
    {
        std::lock_guard<std::mutex> model_lock(model_mutex_);
        require(models_.empty(),"Validate before preparation; reset the controller first");
    }
    backend_->validate_model(states,torques,atol,rtol);
}
bool Controller::has_model(std::int64_t block) const {
    std::lock_guard<std::mutex> lock(model_mutex_);
    return models_.count(block)>0;
}
void Controller::reset() {
    std::scoped_lock lock(prepare_mutex_,feedback_mutex_,model_mutex_);
    models_.clear(); last_result_.reset(); last_tick_=-1;
}
} // namespace mpc_v3
