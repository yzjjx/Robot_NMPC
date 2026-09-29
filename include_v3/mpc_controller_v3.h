#pragma once
#include "prepared_model.h"
#include <map>
#include <mutex>

namespace mpc_v3 {
class Controller {
public:
    Controller(Config config,std::unique_ptr<DynamicsBackend> backend);
    // block=0,1,...，每块覆盖一个model_dt。anchor是该块起点的预测/实测状态。
    PreparationInfo prepare(std::int64_t block,const State& anchor,
                            const Eigen::MatrixXd& state_ref,const Eigen::MatrixXd& ddq_ref);
    // tick=0,1,...，每tick间隔feedback_dt。不会等待缺失的准备结果。
    FeedbackResult feedback(const State& measured,std::int64_t tick,double measurement_age_s=0);
    void validate_backend(const Eigen::MatrixXd& states,const Eigen::MatrixXd& torques,
                          double absolute_tolerance=0.05,double relative_tolerance=0.002);
    bool has_model(std::int64_t block) const;
    void reset();
    Config config() const { return config_; }
    std::string backend_name() const { return backend_name_; }
private:
    const Config config_;
    std::unique_ptr<DynamicsBackend> backend_;
    const std::string backend_name_;
    // 两个长任务用不同锁。model_mutex只用于短时间交换完整快照。
    mutable std::mutex model_mutex_;
    std::mutex prepare_mutex_,feedback_mutex_;
    std::map<std::int64_t,std::shared_ptr<PreparedBlock>> models_;
    std::shared_ptr<const FeedbackResult> last_result_;
    std::int64_t last_tick_=-1;
};
} // namespace mpc_v3
