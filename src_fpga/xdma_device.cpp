#include "aba_device.h"
#include <cerrno>
#include <cstring>
#include <thread>
#include <limits>
#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <sys/stat.h>
#endif

namespace mpc_fpga {
XdmaDevice::XdmaDevice(DeviceOptions options):options_(std::move(options)) {
    require(options_.timeout_ms>0 && options_.poll_us>=0,"Invalid XDMA timeout/poll interval");
    require(options_.base<=static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())-0x20000,
            "AXI base address overflow");
#ifdef __linux__
    h2c_=::open(options_.h2c.c_str(),O_WRONLY|O_CLOEXEC);
    if(h2c_<0) throw std::runtime_error("Cannot open H2C: "+std::string(std::strerror(errno)));
    try {
        struct stat h2c_info{};
        if(::fstat(h2c_,&h2c_info)!=0 || !S_ISCHR(h2c_info.st_mode))
            throw std::runtime_error("H2C path must be an XDMA character device, not a regular file");
        if(::flock(h2c_,LOCK_EX|LOCK_NB)!=0) throw std::runtime_error("Cannot exclusively lock H2C device");
        c2h_=::open(options_.c2h.c_str(),O_RDONLY|O_CLOEXEC);
        if(c2h_<0) throw std::runtime_error("Cannot open C2H: "+std::string(std::strerror(errno)));
        struct stat c2h_info{};
        if(::fstat(c2h_,&c2h_info)!=0 || !S_ISCHR(c2h_info.st_mode))
            throw std::runtime_error("C2H path must be an XDMA character device");
        // 不擅自清除其他任务，也不对板卡执行复位。
        if(read_word(COMMAND)!=0 || read_word(STATUS)!=0)
            throw std::runtime_error("FPGA mailbox is not idle; stop other clients and explicitly recover board state");
    } catch(...) {
        if(c2h_>=0) ::close(c2h_);
        ::close(h2c_); h2c_=c2h_=-1; throw;
    }
#else
    throw std::runtime_error("Actual XDMA access is supported only on Linux");
#endif
}
XdmaDevice::~XdmaDevice() {
#ifdef __linux__
    if(c2h_>=0) ::close(c2h_);
    if(h2c_>=0) ::close(h2c_);
#endif
}
void XdmaDevice::transfer(bool write,std::uint64_t address,void* bytes,std::size_t size) {
#ifdef __linux__
    std::size_t done=0;
    const auto start=Clock::now();
    while(done<size) {
        if(elapsed_ms(start)>options_.timeout_ms) throw std::runtime_error("XDMA transfer loop timeout");
        auto* pointer=static_cast<std::uint8_t*>(bytes)+done;
        const auto offset=static_cast<off_t>(address+done);
        const ssize_t count=write?::pwrite(h2c_,pointer,size-done,offset):
                                  ::pread(c2h_,pointer,size-done,offset);
        if(count<0 && errno==EINTR) continue;
        if(count<=0) throw std::runtime_error("XDMA transfer failed: "+std::string(std::strerror(errno)));
        done+=static_cast<std::size_t>(count);
    }
#else
    (void)write;(void)address;(void)bytes;(void)size;
    throw std::runtime_error("XDMA requires Linux");
#endif
}
std::uint32_t XdmaDevice::read_word(std::uint64_t offset) {
    std::uint8_t b[4]; transfer(false,options_.base+offset,b,4);
    std::uint32_t value=0;
    for(int i=0;i<4;++i) value|=static_cast<std::uint32_t>(b[i])<<(8*i);
    return value;
}
void XdmaDevice::write_word(std::uint64_t offset,std::uint32_t value) {
    std::uint8_t b[4];
    for(int i=0;i<4;++i) b[i]=static_cast<std::uint8_t>(value>>(8*i));
    transfer(true,options_.base+offset,b,4);
}
void XdmaDevice::wait_status(std::uint32_t expected) {
    const auto start=Clock::now();
    while(true) {
        const auto status=read_word(STATUS);
        if(status==expected) return;
        if(status>=4) throw std::runtime_error("FPGA reported error/unknown status");
        if(elapsed_ms(start)>options_.timeout_ms) throw std::runtime_error("FPGA completion timeout");
        if(options_.poll_us) std::this_thread::sleep_for(std::chrono::microseconds(options_.poll_us));
    }
}
std::vector<Joint> XdmaDevice::run(const std::vector<State>& x,const std::vector<Joint>& u) {
    if(poisoned_) throw std::runtime_error("XDMA client poisoned after error; explicit board recovery required");
    auto input=pack_inputs(x,u);
    std::vector<std::uint8_t> output(x.size()*OUTPUT_BYTES);
    const auto start=Clock::now();
    try {
        if(read_word(COMMAND)!=0 || read_word(STATUS)!=0) throw std::runtime_error("FPGA is not idle");
        transfer(true,options_.base+INPUT,input.data(),input.size());
        write_word(COUNT,static_cast<std::uint32_t>(x.size()));
        write_word(COMMAND,1);
        wait_status(3); // 整批完成，无批内抢占、无标签返回、无真正乒乓
        transfer(false,options_.base+OUTPUT,output.data(),output.size());
        auto result=unpack_outputs(output);
        write_word(COMMAND,0);
        wait_status(0);
        ++stats_.batches; stats_.samples+=x.size(); stats_.transfer_and_wait_ms+=elapsed_ms(start);
        return result;
    } catch(...) {
        // 不重发、不复位、不把旧BRAM当新结果；未排空流水线可能仍有在途输出。
        poisoned_=true; throw;
    }
}
} // namespace mpc_fpga
