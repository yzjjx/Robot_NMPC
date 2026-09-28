"""单次 SQP / multiple-shooting MPC，全部预测节点一次批量线性化。

与旧版区别：不逐节点串行非线性 rollout；保留名义轨迹的动力学缺陷项。
每区间采用一次常加速度积分；不是旧版 1 ms 子步积分的等价替换。
"""
from __future__ import annotations

import time
import numpy as np
from scipy import sparse


def linearize_batch(backend, states, controls, dt, cancel_gravity=True,
                    steps=(0.002, 0.002, 0.01)):
    """返回离散 F、A、B 和连续加速度雅可比；每节点 50 或 37 组 ABA。

中心差分顺序：[名义, +18 个扰动, -18 个扰动]。
零重力加速度 = ABA_g(q,dq,u) - ABA_g(q,0,0)。后项只依赖 q，
故每节点仅需 13 组重力基线，避免每个样本在 CPU 上计算重力。
"""
    x, u = np.asarray(states, float), np.asarray(controls, float)
    n = len(x)
    if x.shape != (n, 12) or u.shape != (n, 6) or n < 1:
        raise ValueError("Expected states (N,12), controls (N,6)")
    if not np.isfinite([dt, *steps]).all() or dt <= 0 or min(steps) <= 0:
        raise ValueError("dt and finite-difference steps must be positive")
    eps = np.repeat(steps, 6)
    samples = np.repeat(np.concatenate((x, u), axis=1)[:, None, :], 37, axis=1)
    j = np.arange(18)
    samples[:, 1+j, j] += eps
    samples[:, 19+j, j] -= eps
    payload = samples.reshape(-1, 18)
    if cancel_gravity:
        # q 的 13 个不同取值：名义、+q1..q6、-q1..q6。
        rest = np.zeros((n, 13, 18))
        rest[:, :, :6] = samples[:, np.r_[0, 1:7, 19:25], :6]
        payload = np.concatenate((payload, rest.reshape(-1, 18)))
    if len(payload) > 1000:
        raise ValueError("Batch exceeds BRAM capacity: use horizon <=20 (earth) / <=27 (zero)")
    result = backend.compute(payload)
    if result.shape != (len(payload), 6) or not np.isfinite(result).all():
        raise RuntimeError("Invalid ABA output shape or nonfinite values")
    acc = result[:37*n].reshape(n, 37, 6).copy()
    if cancel_gravity:
        baseline = result[37*n:].reshape(n, 13, 6)
        mapping = np.r_[0, np.arange(1, 7), np.zeros(12, int),
                        np.arange(7, 13), np.zeros(12, int)]
        acc -= baseline[:, mapping, :]
    jac = ((acc[:, 1:19] - acc[:, 19:37]) / (2*eps)[None, :, None]).transpose(0, 2, 1)
    # F=[q+dt*dq+0.5*dt^2*ddq, dq+dt*ddq]；对加速度求导后组装 A、B。
    f = x.copy()
    f[:, :6] += dt*x[:, 6:] + 0.5*dt*dt*acc[:, 0]
    f[:, 6:] += dt*acc[:, 0]
    a = np.tile(np.eye(12), (n, 1, 1))
    a[:, :6, 6:] += dt*np.eye(6)
    a[:, :6] += 0.5*dt*dt*jac[:, :, :12]
    a[:, 6:] += dt*jac[:, :, :12]
    b = np.concatenate((0.5*dt*dt*jac[:, :, 12:], dt*jac[:, :, 12:]), axis=1)
    return f, a, b, jac


class FpgaMPC:
    """变量 z=[x0,...,xN,u0,...,uN-1]；求解器结构只建立一次。"""

    def __init__(self, backend, cpu_zero, dt=0.01, horizon=15, torque_limit=10.0,
                 cancel_gravity=True, fd_steps=(0.002, 0.002, 0.01)):
        import osqp
        if not 1 <= horizon <= (20 if cancel_gravity else 27):
            raise ValueError("Horizon exceeds single-batch capacity")
        if not np.isfinite([dt, torque_limit, *fd_steps]).all() or dt <= 0 or not 0 < torque_limit <= 30:
            raise ValueError("Invalid period/torque limit")
        if min(fd_steps) <= 0 or fd_steps[0] >= 0.1 or fd_steps[1] >= 0.1:
            raise ValueError("Invalid finite-difference steps")
        self.backend, self.cpu = backend, cpu_zero
        self.dt, self.n, self.cancel_gravity = dt, horizon, cancel_gravity
        self.fd_steps = fd_steps
        self.nx = (horizon+1)*12
        self.nz = self.nx + horizon*6
        self.neq = self.nx
        self.xbar = self.ubar = None
        self.solver = osqp.OSQP()
        self.initialized = False
        self.last_timing = {}
        # 收紧到 HLS 源码声明的 q/dq 部署域，给差分扰动保留余量。
        qmin = np.maximum(cpu_zero.model.lowerPositionLimit, -np.pi + 2*fd_steps[0])
        qmax = np.minimum(cpu_zero.model.upperPositionLimit, np.pi - 2*fd_steps[0])
        vmax = np.minimum(cpu_zero.model.velocityLimit, 2.0 - 2*fd_steps[1])
        self.state_min, self.state_max = np.r_[qmin, -vmax], np.r_[qmax, vmax]
        self.limit = torque_limit
        self.box_min = np.r_[np.tile(self.state_min, horizon+1), np.full(horizon*6, -torque_limit)]
        self.box_max = np.r_[np.tile(self.state_max, horizon+1), np.full(horizon*6, torque_limit)]
        weight = np.r_[np.full(6, 100.0), np.ones(6)]
        self.weights = np.r_[np.zeros(12), np.tile(weight, horizon-1), 2*weight,
                             np.full(horizon*6, 0.01)]
        self.p = sparse.diags(2*self.weights + 1e-8, format="csc")
        self._make_constraint_pattern()

    def _make_constraint_pattern(self):
        """保留稠密 A/B 块的显式零；后续更新不改变 CSC 稀疏位置。"""
        rows, cols, vals = list(range(12)), list(range(12)), [1.0]*12
        a_entries, b_entries = [], []
        for k in range(self.n):
            r = 12 + k*12
            for i in range(12):
                rows.append(r+i); cols.append((k+1)*12+i); vals.append(1.0)
                for j in range(12):
                    a_entries.append(len(vals))
                    rows.append(r+i); cols.append(k*12+j); vals.append(0.0)
                for j in range(6):
                    b_entries.append(len(vals))
                    rows.append(r+i); cols.append(self.nx+k*6+j); vals.append(0.0)
        rows.extend(self.neq + np.arange(self.nz))
        cols.extend(range(self.nz)); vals.extend([1.0]*self.nz)
        order = np.lexsort((rows, cols))
        inverse = np.argsort(order)
        self.a_slots, self.b_slots = inverse[a_entries], inverse[b_entries]
        self.constraint = sparse.csc_matrix((vals, (rows, cols)),
                                             shape=(self.neq+self.nz, self.nz))
        self.constraint.sort_indices()

    def compute_control(self, state, state_ref, ddq_ref):
        t0 = time.perf_counter()
        x = np.asarray(state, float)
        ref = np.asarray(state_ref, float)
        ddq_ref = np.asarray(ddq_ref, float)
        if x.shape != (12,) or ref.shape != (self.n+1, 12) or ddq_ref.shape != (self.n, 6):
            raise ValueError("Incorrect state/reference dimensions")
        if not all(np.isfinite(v).all() for v in (x, ref, ddq_ref)):
            raise ValueError("Nonfinite MPC input")
        if np.any(x < self.state_min) or np.any(x > self.state_max):
            raise RuntimeError("Measured state outside joint/FPGA deployment bounds")
        uref = self.cpu.reference_torque(ref[:-1], ddq_ref)
        if self.xbar is None:
            xb = np.clip(ref.copy(), self.state_min, self.state_max)
            ub = np.clip(uref, -self.limit, self.limit)
        else:
            xb = np.vstack((self.xbar[1:], self.xbar[-1]))
            ub = np.vstack((self.ubar[1:], self.ubar[-1]))
        xb[0] = x
        t1 = time.perf_counter()
        f, a, b, _ = linearize_batch(self.backend, xb[:-1], ub, self.dt,
                                     self.cancel_gravity, self.fd_steps)
        t2 = time.perf_counter()
        # 关键：名义轨迹不是精确 rollout，必须保留 c=F-A*xbar-B*ubar。
        c = f - np.einsum("nij,nj->ni", a, xb[:-1]) - np.einsum("nij,nj->ni", b, ub)
        rhs = np.r_[x, c.ravel()]
        low, high = np.r_[rhs, self.box_min], np.r_[rhs, self.box_max]
        target = np.r_[ref.ravel(), uref.ravel()]
        gradient = -2*self.weights*target
        self.constraint.data[self.a_slots] = -a.ravel()
        self.constraint.data[self.b_slots] = -b.ravel()
        if not all(np.isfinite(v).all() for v in (rhs, gradient, self.constraint.data)):
            raise RuntimeError("Nonfinite QP data")
        if not self.initialized:
            self.solver.setup(P=self.p, q=gradient, A=self.constraint, l=low, u=high,
                              verbose=False, warm_starting=True, polishing=False,
                              eps_abs=1e-5, eps_rel=1e-5, max_iter=4000)
            self.initialized = True
        else:
            self.solver.update(q=gradient, l=low, u=high, Ax=self.constraint.data)
        self.solver.warm_start(x=np.r_[xb.ravel(), ub.ravel()])
        t3 = time.perf_counter()
        solution = self.solver.solve(raise_error=False)
        t4 = time.perf_counter()
        if solution.info.status_val != 1 or solution.x is None or not np.isfinite(solution.x).all():
            raise RuntimeError(f"QP failed: {solution.info.status}; simulation stopped")
        z = solution.x
        constraint_value = self.constraint @ z
        violation = max(float(np.max(low-constraint_value)), float(np.max(constraint_value-high)))
        if violation > 2e-3:
            raise RuntimeError(f"QP constraint residual too large: {violation:g}")
        # 仅清除求解公差造成的微小越界；不把失败结果裁剪成有效控制。
        self.xbar = np.clip(z[:self.nx].reshape(self.n+1, 12), self.state_min, self.state_max)
        self.ubar = np.clip(z[self.nx:].reshape(self.n, 6), -self.limit, self.limit)
        self.last_timing = dict(reference_ms=(t1-t0)*1000, linearize_ms=(t2-t1)*1000,
                                qp_update_ms=(t3-t2)*1000, qp_solve_ms=(t4-t3)*1000,
                                total_ms=(time.perf_counter()-t0)*1000,
                                qp_iterations=solution.info.iter,
                                **self.backend.last_timing)
        return self.ubar[0].copy()
