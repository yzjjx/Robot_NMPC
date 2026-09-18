#include "NMPC_control.h"
#include "pinocchio_fun.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

// 把动力学对象和控制器放在一起，保证 dynamics 的生命周期足够长。
class MPCController {
    pinocchioFun dynamics;
    ROKAE_NMPC controller;

public:
    explicit MPCController(const std::string& urdf_path, double timestep, int horizon)
        : dynamics(urdf_path), controller(dynamics, timestep, horizon) {}

    double timestep() const { return controller.timestep(); }
    int horizon() const { return controller.horizon(); }

    Eigen::VectorXd predict_state(const Eigen::VectorXd& state,
                                 const Eigen::VectorXd& tau, double duration)
    {
        if(state.size() != 12 || tau.size() != 6 || !state.allFinite() || !tau.allFinite()) {
            throw std::invalid_argument("Expected finite state (12,) and torque (6,).");
        }
        return dynamics.compute_held_state(state, tau, duration);
    }

    Eigen::VectorXd compute_control(
        const Eigen::VectorXd& state,
        const Eigen::MatrixXd& state_ref,
        const Eigen::MatrixXd& ddq_ref)
    {
        const int N = horizon();
        // NumPy 的每一行对应一个预测时刻，避免错误输入导致 C++ 越界。
        if(state.size() != 12 || state_ref.rows() != N+1 ||
           state_ref.cols() != 12 || ddq_ref.rows() != N || ddq_ref.cols() != 6) {
            throw std::invalid_argument(
                "Expected state (12,), state_ref (N+1, 12), ddq_ref (N, 6).");
        }
        if(!state.allFinite() || !state_ref.allFinite() || !ddq_ref.allFinite()) {
            throw std::invalid_argument("Controller inputs must be finite.");
        }

        std::vector<Eigen::VectorXd> states(N+1), accelerations(N);
        for(int i = 0; i <= N; ++i) states[i] = state_ref.row(i).transpose();
        for(int i = 0; i < N; ++i) accelerations[i] = ddq_ref.row(i).transpose();

        Eigen::VectorXd tau = controller.compute_control(state, states, accelerations);
        // 原控制器失败时返回名义力矩；评估时必须明确报错，不能当成成功的 MPC。
        if(!controller.qp_success() || !tau.allFinite()) {
            throw std::runtime_error("MPC QP failed; tracking experiment stopped.");
        }
        return tau;
    }
};

PYBIND11_MODULE(rokae_mpc, module) {
    module.doc() = "ROKAE SR4 C++ MPC controller";
    py::class_<MPCController>(module, "MPCController")
        .def(py::init<const std::string&, double, int>(), py::arg("urdf_path"),
             py::arg("timestep") = 0.001, py::arg("horizon") = 40)
        .def_property_readonly("timestep", &MPCController::timestep)
        .def_property_readonly("horizon", &MPCController::horizon)
        .def("predict_state", &MPCController::predict_state,
             py::arg("state"), py::arg("tau"), py::arg("duration"),
             py::call_guard<py::gil_scoped_release>())
        .def("compute_control", &MPCController::compute_control,
             py::arg("state"), py::arg("state_ref"), py::arg("ddq_ref"),
             py::call_guard<py::gil_scoped_release>());
}
