"""CPU/mock/FPGA准备-反馈对照。只运行Pinocchio虚拟对象，不发送机器人力矩。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
from pathlib import Path
import time

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parent.parent


def validation_samples(center):
    """中心点 + 每个q/dq/tau维度正负扰动，仅是起步检查，不覆盖整个工作空间。"""
    states = np.tile(center, (37, 1))
    torques = np.zeros((37, 6))
    for j in range(18):
        if j < 12:
            states[1 + 2*j, j] += 0.02
            states[2 + 2*j, j] -= 0.02
        else:
            torques[1 + 2*j, j-12] += 0.1
            torques[2 + 2*j, j-12] -= 0.1
    return states, torques


def reference(times):
    """小幅平滑测试轨迹：关节1为0.05*sin(t)，其余为零。"""
    states = np.zeros((len(times), 12))
    ddq = np.zeros((len(times), 6))
    states[:, 0] = 0.05*np.sin(times)
    states[:, 6] = 0.05*np.cos(times)
    ddq[:, 0] = -0.05*np.sin(times)
    return states, ddq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "mock", "fpga"), default="cpu")
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--mode", choices=("numerical", "paced"), default="numerical")
    parser.add_argument("--urdf", type=Path, default=ROOT / "urdf/ROKAE_SR4.urdf")
    parser.add_argument("--model-dt", type=float, default=0.01)
    parser.add_argument("--ratio", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be finite and positive")
    if args.backend == "fpga" and not args.allow_hardware:
        parser.error("actual FPGA access requires --allow-hardware")
    module = importlib.import_module("rokae_mpc_v3" if args.backend == "cpu" else "rokae_mpc_fpga")
    config = module.Config()
    config.model_dt, config.feedback_ratio, config.horizon = args.model_dt, args.ratio, args.horizon
    config.torque_limit = 10.0  # 三种后端相同，不把限幅变化算成硬件收益。
    config.enforce_feedback_budget = args.mode == "paced"
    mpc = module.MPCController(str(args.urdf), config, backend=args.backend,
                               allow_hardware=args.allow_hardware)
    dt = config.feedback_dt
    ticks = int(args.duration/dt)
    if ticks < 1:
        parser.error("duration shorter than one feedback tick")
    plant = pin.buildModelFromUrdf(str(args.urdf))
    if (plant.nq, plant.nv) != (6, 6):
        raise ValueError("Expected 6-DOF plant")
    plant.gravity.setZero()  # 与控制器默认附加力矩约定一致。
    plant_data = plant.createData()  # 与准备线程完全独立的Data

    def advance(state, torque):
        steps = max(1, int(np.ceil(dt/config.integration_step)))
        h = dt/steps
        result = state.copy()
        for _ in range(steps):
            acceleration = np.asarray(pin.aba(plant, plant_data, result[:6], result[6:], torque)).copy()
            result[:6] += h*result[6:] + 0.5*h*h*acceleration
            result[6:] += h*acceleration
        if not np.isfinite(result).all():
            raise RuntimeError("Virtual plant became nonfinite")
        return result

    def refs_for(block):
        return reference((block+np.arange(config.horizon+1))*config.model_dt)

    initial, _ = reference(np.array([0.0]))
    state = initial[0]
    print(f"backend={mpc.backend_name}; mode={args.mode}; "
          f"target feedback={1/dt:.3f} Hz; model={1/config.model_dt:.3f} Hz")
    print("No robot commands are sent. Mock timings are NOT FPGA timings.")
    if args.backend != "cpu":
        vx, vu = validation_samples(state)
        mpc.validate_backend(vx, vu)  # 不匹配会停止，没有忽略检查的开关。
    refs, ddq = refs_for(0)
    first = mpc.prepare(0, state, refs, ddq[:-1])
    preparation_rows = [[0, first.total_ms, first.model_ms, first.matrices_and_qp_ms,
                         first.backend.batches, first.backend.samples]]
    rows = []
    missed_preparation = 0
    failure = None
    # 第一个模型在计时前准备。首块冷启动耗时单独记录，不能当作零成本。
    wall_start = time.perf_counter()
    pending = None
    pending_block = None
    with ThreadPoolExecutor(max_workers=1) as worker:
        try:
            for tick in range(ticks):
                target = wall_start+tick*dt
                if args.mode == "paced":
                    time.sleep(max(0.0, target-time.perf_counter()))
                block, phase = divmod(tick, config.feedback_ratio)
                if pending is not None and pending_block == block and phase == 0:
                    if not pending.done():
                        missed_preparation += 1
                        if args.mode == "paced":
                            raise RuntimeError("Next model missed its deadline; stopped without stale-model fallback")
                    info = pending.result()  # numerical模式可等待，不能据此宣称实时。
                    preparation_rows.append([info.block, info.total_ms, info.model_ms,
                                             info.matrices_and_qp_ms, info.backend.batches, info.backend.samples])
                    pending = None
                sampled = time.perf_counter()
                result = mpc.feedback(state.copy(), tick, time.perf_counter()-sampled)
                completed = time.perf_counter()
                wall_feedback_ms = 1000*(completed-sampled)
                actual_deadline_miss = args.mode == "paced" and completed > target+dt
                if actual_deadline_miss:
                    raise RuntimeError("Wall-clock feedback deadline missed; result not applied")
                rows.append(np.r_[tick, tick*dt, wall_feedback_ms, result.solve_ms,
                                  result.total_ms, result.qp_iterations, state, result.torque])
                # 本周期第一个反馈完成后，提前准备下一模型块。
                if phase == 0 and (block+1)*config.feedback_ratio < ticks:
                    next_refs, next_ddq = refs_for(block+1)
                    predicted_anchor = result.states[1].copy()
                    pending_block = block+1
                    pending = worker.submit(mpc.prepare, pending_block, predicted_anchor,
                                            next_refs, next_ddq[:-1])
                state = advance(state, result.torque)
            if pending is not None:
                pending.result()  # 不静默丢掉后台异常
        except Exception as error:
            failure = error
    elapsed = time.perf_counter()-wall_start
    output = args.output or Path(__file__).resolve().parent / "results" / f"{args.backend}_{args.mode}"
    output.mkdir(parents=True, exist_ok=True)
    if rows:
        data = np.asarray(rows)
        header = "tick,time,feedback_wall_ms,qp_ms,cpp_ms,qp_iterations," + ",".join(
            f"{kind}{j}" for kind in ("q", "dq", "tau") for j in range(1, 7))
        np.savetxt(output/"feedback.csv", data, delimiter=",", header=header, comments="")
        desired, _ = reference(data[:, 1])
        print("q RMSE [rad]:", np.sqrt(np.mean((data[:, 6:12]-desired[:, :6])**2, axis=0)))
        print(f"feedback mean/P99/max [ms]: {data[:,2].mean():.3f} / "
              f"{np.percentile(data[:,2],99):.3f} / {data[:,2].max():.3f}")
    np.savetxt(output/"preparation.csv", np.asarray(preparation_rows), delimiter=",",
               header="block,total_ms,model_ms,matrices_qp_ms,batches,samples", comments="")
    print(f"completed ticks={len(rows)}/{ticks}, elapsed={elapsed:.3f}s, "
          f"observed preparation misses={missed_preparation}, output={output}")
    print("This is a virtual plant. Numerical mode pauses as needed; paced mode tests host scheduling,")
    print("but neither models actual robot communication nor certifies hardware closed-loop stability.")
    if failure is not None:
        raise RuntimeError(f"Experiment stopped: {failure}") from failure


if __name__ == "__main__":
    main()
