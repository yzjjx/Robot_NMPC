#pragma once
#include "../include_v3/mpc_types_v3.h"
#include <memory>
#include <utility>

namespace mpc_fpga {
using namespace mpc_v3;
struct DeviceOptions {
    std::string h2c="/dev/xdma0_h2c_0", c2h="/dev/xdma0_c2h_0";
    std::uint64_t base=0xC0000000ULL;
    int timeout_ms=100, poll_us=10;
};

// 与现有bit一致，不能向这里添加task_id或改变数据排列。
constexpr std::uint64_t COMMAND=0x80,STATUS=0x84,COUNT=0x88,INPUT=0x100,OUTPUT=0x12000;
constexpr std::size_t MAX_BATCH=1000, INPUT_BYTES=72, OUTPUT_BYTES=24;
std::int32_t quantize(double value,int fractional_bits);
void check_domain(const State& x,const Joint& u);
std::vector<std::uint8_t> pack_inputs(const std::vector<State>& x,const std::vector<Joint>& u);
std::vector<Joint> unpack_outputs(const std::vector<std::uint8_t>& bytes);
std::pair<State,Joint> quantized_input(const State& x,const Joint& u);

class AbaDevice {
public:
    virtual ~AbaDevice()=default;
    virtual std::vector<Joint> run(const std::vector<State>& x,const std::vector<Joint>& u)=0;
    virtual BackendStats stats() const=0;
    virtual void clear_stats()=0;
    virtual std::string name() const=0;
};

class XdmaDevice final : public AbaDevice {
public:
    explicit XdmaDevice(DeviceOptions options);
    ~XdmaDevice() override;
    XdmaDevice(const XdmaDevice&)=delete;
    XdmaDevice& operator=(const XdmaDevice&)=delete;
    std::vector<Joint> run(const std::vector<State>& x,const std::vector<Joint>& u) override;
    BackendStats stats() const override { return stats_; }
    void clear_stats() override { stats_={}; }
    std::string name() const override { return "xdma_hardware"; }
private:
    DeviceOptions options_;
    int h2c_=-1,c2h_=-1;
    bool poisoned_=false;
    BackendStats stats_;
    void transfer(bool write,std::uint64_t address,void* bytes,std::size_t size);
    std::uint32_t read_word(std::uint64_t offset);
    void write_word(std::uint64_t offset,std::uint32_t value);
    void wait_status(std::uint32_t expected);
};

// 离线验证调度用：量化后的Pinocchio，不模拟真实HLS内部误差，也不能测硬件速度。
class MockDevice final : public AbaDevice {
public:
    MockDevice(const std::string& urdf,const Config& c):dynamics_(urdf,c.gravity_compensated,c.integration_step) {}
    std::vector<Joint> run(const std::vector<State>& x,const std::vector<Joint>& u) override;
    BackendStats stats() const override { return stats_; }
    void clear_stats() override { stats_={}; }
    std::string name() const override { return "mock_not_hardware"; }
private:
    mpc_v2::Dynamics dynamics_;
    BackendStats stats_;
};
} // namespace mpc_fpga
