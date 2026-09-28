"""同一组状态快照比较旧版与 v2。不会连接机器人，不写旧工程文件。"""
import argparse
from pathlib import Path
import time

import numpy as np
from rokae_mpc_v2 import MPCController

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=ROOT / "urdf/ROKAE_SR4.urdf")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--old", action="store_true", help="同时导入旧 rokae_mpc 模块")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    controllers = [("v2", MPCController(str(args.urdf), args.dt, args.horizon))]
    if args.old:
        from rokae_mpc import MPCController as OldController
        controllers.insert(0, ("old", OldController(str(args.urdf), args.dt, args.horizon)))
    reference = np.zeros((args.horizon + 1, 12))
    ddq = np.zeros((args.horizon, 6))
    results = {}
    for name, controller in controllers:
        times, torques = [], []
        for k in range(args.count + 5):
            state = np.zeros(12)
            state[0] = 0.01 * np.sin(0.1 * k)
            start = time.perf_counter()
            torque = controller.compute_control(state, reference, ddq)
            elapsed_ms = 1000 * (time.perf_counter() - start)
            if k >= 5:
                times.append(elapsed_ms)
                torques.append(torque)
        results[name] = np.asarray(torques)
        print(f"{name}: mean={np.mean(times):.3f} ms, "
              f"P99={np.percentile(times, 99):.3f} ms, max={np.max(times):.3f} ms, "
              f"budget misses={np.count_nonzero(np.asarray(times) >= 1000*args.dt)}/{args.count}")
    if args.old:
        print("max old/v2 torque difference [Nm]:", np.max(np.abs(results["old"] - results["v2"])))
    print("Snapshot benchmark only: not measured real-robot control frequency.")


if __name__ == "__main__":
    main()
