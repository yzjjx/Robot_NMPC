# 当前ABA IP的异构准备/反馈 MPC

先阅读 `../src_v3/README.md`。本目录与 `include_fpga` 提供动力学后端，
**控制数学、相位对齐、缺陷修正、快速反馈全部复用v3**，便于对照。
原有工程和bit不变。本次未编译、未访问设备、未执行预检查或实物实验。

## 1. 实际做了什么

准备线程：

1. 已知多个预测区间的起点，将当前小步的q/dq/tau组成批次。
2. 通过常驻XDMA文件描述符写BRAM，启动现有ABA核。
3. 整批完成后读取ddq，在CPU上积分。
4. CPU计算平滑Pinocchio模型的小步解析导数并递推敏感度。
5. 下一轮发送各条积分链的新状态，最后构造模型和QP并交给反馈线程。

反馈线程只执行v3的`feedback()`，不访问FPGA。

**当前是近似雅可比的混合模型：**状态传播使用FPGA定点加速度，导数来自CPU浮点模型。
定点舍入本身并不光滑，所以这不是对量化程序逐位求导。
必须先确认硬件与CPU动力学足够一致，再验证导数、闭环误差和实际收益。

CPU解析导数接口仍含动力学计算，本版并未消除这部分成本，也没有专门的导数IP。
多小步仍需多轮PCIe往返；因此不能保证FPGA准备比纯CPU更快，更不能先承诺三倍。
必须分别比较：v2、v3(cpu)、v3(fpga)。准备/反馈分离带来的收益不能全算成硬件收益。

## 2. 匹配当前硬件的接口

| 项目 | 值 |
|---|---|
| XDMA AXI基址 | `0xC0000000` |
| command/status/count | `+0x80 / +0x84 / +0x88` |
| 输入地址 | `+0x100` |
| 输出地址 | `+0x12000` |
| 每组输入 | 72字节，q[6]、dq[6]、tau[6] |
| 每组输出 | 24字节，ddq[6] |
| 格式 | 有符号int32，小端；q/dq Q4.28，tau Q8.24，ddq Q16.16 |
| 批量 | 1..1000，超过时由软件分批 |

没有添加task_id，没有假装支持批内抢占/结果提前回收/真正乒乓。
样本按批内顺序映射回对应预测节点和积分小步。
正常流程为：检查idle -> 写输入/count -> command=1 -> status=3 -> 读结果 -> command=0 -> idle。
发生错误/超时后客户端被标记为不可继续使用；不会自动重发、复位或使用旧BRAM。
用户必须检查板卡状态并显式恢复。板上在途结果未排空时，盲目重新启动可能串批。

代码同时检查当前HLS部署域：q在[-pi,pi]、dq在[-2,2]、tau在[-10,10]。
它们比固定点字长的可表示范围更窄。超域/饱和直接拒绝，不悄悄截断。
输出没有硬件逐样本溢出标志，主机检测极限编码并不保证能发现所有内部定点溢出。

`XdmaDevice`仅支持Linux实际访问。使用设备文件advisory lock；其他不遵守锁的脚本仍可能干扰，
运行时必须停止其他XDMA客户端。阻塞pread/pwrite若卡在驱动内，用户态超时不能强制中断，
因此这里的超时检查不构成硬实时保证。

## 3. 文件

| 文件 | 职责 |
|---|---|
| `aba_codec.cpp` / `include_fpga/aba_device.h` | 定点打包解包、部署域检查、mock设备 |
| `xdma_device.cpp` | Linux实际XDMA读写和mailbox协议 |
| `fpga_backend.cpp` / `include_fpga/fpga_backend.h` | 批量积分、CPU导数、数值一致性门禁 |
| `python_bindings.cpp` | rokae_mpc_fpga模块，默认mock，硬件访问需显式允许 |
| `test_fpga.cpp` | 格式、mock积分、拒绝错误模型等测试，不打开真实设备 |

## 4. Ubuntu构建（你手动执行）

在项目根目录运行。它同时构建CPU v3和FPGA模块：

```bash
cmake -S src_fpga -B build_fpga_v3 -DCMAKE_BUILD_TYPE=Release
cmake --build build_fpga_v3 -j2
ctest --test-dir build_fpga_v3 --output-on-failure
```

输出模块位于`build_fpga_v3/python`。不会覆盖`python_code`已有模块。
mock测试必须先通过。测试代码已经提供，但本次没有执行。

## 5. 先mock，再真实板卡

不访问板卡的路径：

```bash
PYTHONPATH="$PWD/build_fpga_v3/python" python3 src_v3/demo_pipeline.py --backend mock --mode numerical
```

mock用量化输入和输出的Pinocchio代替ABA设备；用于检查接口和调度，
不模拟HLS全部内部定点格式，不能当作FPGA数值精度/性能报告。

只有确认bit格式、模型和部署域一致后，才显式允许硬件：

```bash
PYTHONPATH="$PWD/build_fpga_v3/python" python3 src_v3/demo_pipeline.py --backend fpga --allow-hardware --mode numerical
```

设备权限应由你的已有Ubuntu环境配置；需要sudo时，使用正确环境的Python绝对路径和显式PYTHONPATH。
这条命令会访问FPGA BRAM并计算ABA，但**不会向机器人发送任何力矩**。
完成数值比较之后，再尝试`--mode paced`；错过模型/反馈截止时间会报错停止。

## 6. 强制模型检查

默认必须先执行`validate_backend(states,torques)`，否则prepare拒绝运行。
检查至少19组样本，并要求每个q/dq/tau维度有变化；示例使用37组正负扰动。
比较CPU在相同量化输入上的ddq与设备输出，条件为：

```text
abs(ddq_fpga - ddq_cpu) <= absolute_tolerance + relative_tolerance*abs(ddq_cpu)
```

默认绝对容差0.05 rad/s²、相对容差0.002只是初始数值门限，不是闭环稳定性界限。
检查只证明这些样本上的一致性，必须扩充为覆盖你的轨迹、速度、力矩和负载的测试集。
还需检查积分后状态、雅可比近似和闭环误差，不能仅凭一次预检查就接实物。

若失败，重点排查：

- CPU是否关闭重力，而bit仍包含重力；不能只为通过门禁随便切换。
- 惯量、质心、关节轴、坐标约定、关节顺序、工具负载是否一致。
- 下载的bit是否与当前定点数据格式匹配。
- 是否错误把float32按int32传输，或使用错误地址。

**没有忽略数值门禁、自动猜重力或自动补偿模型差异的开关。**
显著模型差异时必须先修正模型/bit，而不是无限放大容差。

## 7. 最小调用

```python
from rokae_mpc_fpga import Config, MPCController
cfg = Config()
mpc = MPCController(urdf_path, cfg, backend="fpga", allow_hardware=True)
mpc.validate_backend(representative_states, representative_torques)
mpc.prepare(0, predicted_initial_state, reference_states, reference_accelerations)
result = mpc.feedback(latest_state, tick=0, measurement_age_s=actual_measurement_age)
```

使用`backend="mock"`不需要`allow_hardware=True`。创建模块/导入本身不会打开板卡。
完整的常驻准备线程示例共用`src_v3/demo_pipeline.py`。
当前实时任务不存在硬件P0抢占；过期/失效模型直接拒绝，由调用层执行经验证的回退。
不要把此实验控制器直接接实物而不补齐设备安全层。
