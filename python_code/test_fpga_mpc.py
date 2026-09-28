"""无需板卡、Pinocchio、MuJoCo 的软件测试：python -B -m unittest test_fpga_mpc -v。"""
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fpga_aba import XdmaABA, pack_inputs, unpack_outputs
from fpga_mpc import FpgaMPC, linearize_batch


class ToyABA:
    def __init__(self, gravity=True):
        self.gravity = gravity
        self.last_timing = {}
        self.model = SimpleNamespace(lowerPositionLimit=np.full(6, -3.0),
                                     upperPositionLimit=np.full(6, 3.0),
                                     velocityLimit=np.full(6, 2.0))
        self.batches = []

    def compute(self, samples):
        self.batches.append(samples.copy())
        q, v, u = samples[:, :6], samples[:, 6:12], samples[:, 12:]
        return (1+0.1*np.cos(q))*u - 0.2*v - (np.sin(q) if self.gravity else 0)

    def reference_torque(self, states, accelerations):
        return (accelerations + 0.2*states[:, 6:]) / (1+0.1*np.cos(states[:, :6]))


class FakeMailbox(XdmaABA):
    """模拟寄存器，不打开任何设备。"""
    def __init__(self):
        self.failed = False
        self.timeout, self.poll_interval = 0.001, 0
        self.status, self.count = 0, 0
        self.writes = []
        self.last_timing = {}

    def _status(self):
        return self.status

    def _write(self, offset, payload):
        self.writes.append(offset)
        if offset == 0x88:
            self.count = struct.unpack("<I", payload)[0]
        elif offset == 0x100:
            self.payload = payload
        elif offset == 0x80:
            self.status = 3 if struct.unpack("<I", payload)[0] else 0

    def _read(self, offset, size):
        assert offset == 0x12000 and size == self.count*24
        return np.full((self.count, 6), 65536, dtype="<i4").tobytes()


class TestProtocol(unittest.TestCase):
    def test_encoding_roundtrip(self):
        x = np.tile(np.r_[np.full(6, -0.5), np.full(6, 0.25), np.full(6, -2.0)], (3, 1))
        raw = np.frombuffer(pack_inputs(x), dtype="<i4").reshape(3, 18)
        np.testing.assert_array_equal(raw[0, :6], np.full(6, -2**27))
        np.testing.assert_array_equal(raw[0, 6:12], np.full(6, 2**26))
        np.testing.assert_array_equal(raw[0, 12:], np.full(6, -2**25))
        payload = np.array([[65536, -65536, 0, 32768, 1, -1]], dtype="<i4").tobytes()
        np.testing.assert_array_equal(unpack_outputs(payload, 1),
                                      [[1, -1, 0, 0.5, 2**-16, -2**-16]])

    def test_reject_invalid_input(self):
        for rows in (0, 1001):
            with self.assertRaises(ValueError):
                pack_inputs(np.zeros((rows, 18)))
        for column, value in ((0, 8), (12, 128), (3, np.nan), (4, np.inf)):
            sample = np.zeros((1, 18)); sample[0, column] = value
            with self.assertRaises(ValueError):
                pack_inputs(sample)
        self.assertEqual(len(pack_inputs(np.zeros((1000, 18)))), 72000)

    def test_saturation_and_short_output(self):
        with self.assertRaises(RuntimeError):
            unpack_outputs(b"", 1)
        raw = np.zeros((1, 6), dtype="<i4"); raw[0, 0] = 2**31-1
        with self.assertRaises(RuntimeError):
            unpack_outputs(raw.tobytes(), 1)

    def test_mailbox_two_runs(self):
        board = FakeMailbox()
        for count in (750, 1000):
            np.testing.assert_array_equal(board.compute(np.zeros((count, 18))), np.ones((count, 6)))
            self.assertEqual(board.status, 0)
        self.assertEqual(board.writes, [0x100, 0x88, 0x80, 0x80]*2)

    def test_busy_does_not_overwrite(self):
        board = FakeMailbox(); board.status = 2
        with self.assertRaises(RuntimeError):
            board.compute(np.zeros((1, 18)))
        self.assertEqual(board.writes, [])
        self.assertTrue(board.failed)
        board.status = 0
        with self.assertRaises(RuntimeError):
            board.compute(np.zeros((1, 18)))

    def test_timeout_no_automatic_cancel(self):
        board = FakeMailbox(); board.status = 2
        with self.assertRaises(TimeoutError):
            board._wait(3)
        self.assertEqual(board.writes, [])

    def test_error_status(self):
        board = FakeMailbox(); board.status = 4
        with self.assertRaises(RuntimeError):
            board._wait(3)

    def test_partial_transfers(self):
        board = XdmaABA.__new__(XdmaABA)
        board.base, board.h2c, board.c2h = 0xC0000000, 11, 12
        with patch("os.pwrite", side_effect=[2, 2], create=True) as write:
            board._write(0x88, b"abcd")
            self.assertEqual(write.call_args_list[1].args[2], 0xC000008A)
        with patch("os.pread", side_effect=[b"ab", b"cd"], create=True) as read:
            self.assertEqual(board._read(0x84, 4), b"abcd")
            self.assertEqual(read.call_args_list[1].args[2], 0xC0000086)


class TestLinearization(unittest.TestCase):
    def test_batch_jacobian_and_gravity(self):
        rng = np.random.default_rng(8)
        x, u = rng.uniform(-0.5, 0.5, (15, 12)), rng.uniform(-1, 1, (15, 6))
        board = ToyABA()
        f, a, b, jac = linearize_batch(board, x, u, 0.01)
        self.assertEqual(board.batches[0].shape, (750, 18))
        expected = np.zeros((15, 6, 18))
        j = np.arange(6)
        expected[:, j, j] = -0.1*np.sin(x[:, :6])*u
        expected[:, j, j+6] = -0.2
        expected[:, j, j+12] = 1+0.1*np.cos(x[:, :6])
        np.testing.assert_allclose(jac, expected, atol=1e-7)
        cpu = ToyABA(gravity=False)
        f0, a0, b0, _ = linearize_batch(cpu, x, u, 0.01, False)
        np.testing.assert_allclose(f, f0, atol=1e-12)
        np.testing.assert_allclose(a, a0, atol=1e-12)
        np.testing.assert_allclose(b, b0, atol=1e-12)
        # 对整个离散映射数值求导，验证 q/dq 两半和 dt 系数。
        eps = 1e-5
        for col in range(18):
            xp, up = x.copy(), u.copy()
            if col < 12:
                xp[:, col] += eps
            else:
                up[:, col-12] += eps
            fp, _, _, _ = linearize_batch(cpu, xp, up, 0.01, False)
            np.testing.assert_allclose((fp-f)/eps, a[:, :, col] if col < 12 else b[:, :, col-12], atol=1e-7)

    def test_capacity(self):
        with self.assertRaises(ValueError):
            linearize_batch(ToyABA(), np.zeros((21, 12)), np.zeros((21, 6)), .01)
        backend = ToyABA()
        linearize_batch(backend, np.zeros((20, 12)), np.zeros((20, 6)), .01)
        self.assertEqual(len(backend.batches[0]), 1000)


class TestQP(unittest.TestCase):
    def test_single_stage_analytic_solution(self):
        mpc = FpgaMPC(ToyABA(), ToyABA(False), dt=.01, horizon=1)
        x = np.zeros(12)
        ref = np.zeros((2, 12)); ref[1, :6] = .2
        actual = mpc.compute_control(x, ref, np.zeros((1, 6)))
        bq, bv = .5*.01**2*1.1, .01*1.1
        expected = 200*bq*.2 / (.01 + 200*bq*bq + 2*bv*bv)
        np.testing.assert_allclose(actual, np.full(6, expected), atol=2e-4)

    def test_shift_update_and_dynamics_defect(self):
        backend, cpu = ToyABA(), ToyABA(False)
        mpc = FpgaMPC(backend, cpu, dt=.01, horizon=5)
        state = np.zeros(12)
        ref = np.zeros((6, 12)); ref[:, :6] = .1
        indices, indptr = mpc.constraint.indices.copy(), mpc.constraint.indptr.copy()
        for _ in range(30):
            previous = state.copy()
            torque = mpc.compute_control(state, ref, np.zeros((5, 6)))
            self.assertTrue(np.isfinite(torque).all())
            self.assertLessEqual(np.abs(torque).max(), 10)
            np.testing.assert_allclose(mpc.xbar[0], previous, atol=2e-4)
            acceleration = cpu.compute(np.r_[state, torque][None, :])[0]
            state[:6] += .01*state[6:] + .5*.01**2*acceleration
            state[6:] += .01*acceleration
        self.assertGreater(state[0], 0)
        np.testing.assert_array_equal(mpc.constraint.indices, indices)
        np.testing.assert_array_equal(mpc.constraint.indptr, indptr)
        self.assertEqual(len(backend.batches), 30)

    def test_invalid_state_and_solver_failure(self):
        mpc = FpgaMPC(ToyABA(), ToyABA(False), dt=.01, horizon=2)
        state = np.zeros(12); state[0] = 8
        with self.assertRaises(RuntimeError):
            mpc.compute_control(state, np.zeros((3, 12)), np.zeros((2, 6)))
        with patch.object(mpc.solver, "solve", return_value=SimpleNamespace(
                info=SimpleNamespace(status_val=7, status="maximum iterations"), x=None)):
            with self.assertRaises(RuntimeError):
                mpc.compute_control(np.zeros(12), np.zeros((3, 12)), np.zeros((2, 6)))


if __name__ == "__main__":
    unittest.main()
