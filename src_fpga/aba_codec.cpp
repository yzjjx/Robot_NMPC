#include "aba_device.h"
#include <limits>

namespace mpc_fpga {
std::int32_t quantize(double value,int bits) {
    require(std::isfinite(value) && bits>=0 && bits<=30,"Invalid fixed-point input");
    const double scaled=std::ldexp(value,bits);
    const double low=std::floor(scaled),fraction=scaled-low;
    // 确定性的round-to-nearest-even，不依赖进程浮点舍入模式。
    const double rounded=(fraction<0.5)?low:(fraction>0.5)?low+1:
        (std::fmod(std::abs(low),2.0)==0?low:low+1);
    require(rounded>=std::numeric_limits<std::int32_t>::min() &&
            rounded<=std::numeric_limits<std::int32_t>::max(),"Fixed-point input would saturate");
    return static_cast<std::int32_t>(rounded);
}
void check_domain(const State& x,const Joint& u) {
    require(x.allFinite() && u.allFinite(),"Nonfinite FPGA input");
    // 来自当前HLS部署约定；比32位编码的可表示范围更严格。
    require(x.head<NU>().cwiseAbs().maxCoeff()<=3.14159265358979323846,
            "FPGA q outside validated [-pi,pi] domain");
    require(x.tail<NU>().cwiseAbs().maxCoeff()<=2.0,"FPGA dq outside validated [-2,2] domain");
    require(u.cwiseAbs().maxCoeff()<=10.0+1e-8,"FPGA torque outside validated [-10,10] domain");
}
std::pair<State,Joint> quantized_input(const State& x,const Joint& u) {
    check_domain(x,u); State qx; Joint qu;
    for(int j=0;j<NX;++j) qx(j)=std::ldexp(static_cast<double>(quantize(x(j),28)),-28);
    for(int j=0;j<NU;++j) qu(j)=std::ldexp(static_cast<double>(quantize(u(j),24)),-24);
    return {qx,qu};
}
std::vector<std::uint8_t> pack_inputs(const std::vector<State>& x,const std::vector<Joint>& u) {
    require(!x.empty() && x.size()<=MAX_BATCH && x.size()==u.size(),"FPGA batch must contain 1..1000 samples");
    std::vector<std::uint8_t> bytes(x.size()*INPUT_BYTES);
    for(std::size_t i=0;i<x.size();++i) {
        check_domain(x[i],u[i]);
        for(int j=0;j<18;++j) {
            const auto word=static_cast<std::uint32_t>(j<NX?quantize(x[i](j),28):quantize(u[i](j-NX),24));
            for(int b=0;b<4;++b) bytes[i*INPUT_BYTES+j*4+b]=static_cast<std::uint8_t>(word>>(8*b));
        }
    }
    return bytes;
}
std::vector<Joint> unpack_outputs(const std::vector<std::uint8_t>& bytes) {
    require(!bytes.empty() && bytes.size()%OUTPUT_BYTES==0 && bytes.size()/OUTPUT_BYTES<=MAX_BATCH,
            "Invalid FPGA output byte count");
    std::vector<Joint> result(bytes.size()/OUTPUT_BYTES);
    for(std::size_t i=0;i<result.size();++i) for(int j=0;j<NU;++j) {
        std::uint32_t word=0;
        for(int b=0;b<4;++b) word|=static_cast<std::uint32_t>(bytes[i*OUTPUT_BYTES+j*4+b])<<(8*b);
        const std::int64_t signed_word=(word&0x80000000U)?static_cast<std::int64_t>(word)-0x100000000LL:word;
        if(signed_word==std::numeric_limits<std::int32_t>::min() || signed_word==std::numeric_limits<std::int32_t>::max())
            throw std::runtime_error("FPGA output saturated; result rejected");
        result[i](j)=std::ldexp(static_cast<double>(signed_word),-16);
    }
    return result;
}
std::vector<Joint> MockDevice::run(const std::vector<State>& x,const std::vector<Joint>& u) {
    pack_inputs(x,u); // 同样执行输入形状、范围和格式检查
    const auto start=Clock::now();
    std::vector<Joint> result;
    for(std::size_t i=0;i<x.size();++i) {
        const auto in=quantized_input(x[i],u[i]);
        Joint ddq=dynamics_.acceleration(in.first,in.second);
        for(int j=0;j<NU;++j) ddq(j)=std::ldexp(static_cast<double>(quantize(ddq(j),16)),-16);
        result.push_back(ddq);
    }
    ++stats_.batches; stats_.samples+=x.size(); stats_.transfer_and_wait_ms+=elapsed_ms(start);
    return result;
}
} // namespace mpc_fpga
