#include "NMPC_control.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <mutex>
#include <tuple>

namespace py = pybind11;
using namespace mpc_v2;

// Python 对象自己拥有 Dynamics，避免引用已析构对象。
// 锁用于防止调用方误将同一个实例同时用于预测和求解，不是并行加速。
class MPCController {
public:
    MPCController(const std::string& urdf, double timestep, int horizon,
                  bool gravity_compensated, double integration_step, bool warm_start)
        : dynamics_(urdf, gravity_compensated, integration_step),
          controller_(dynamics_, timestep, horizon, warm_start) {}

    Joint compute_control(const State& state, const Eigen::MatrixXd& refs,
                          const Eigen::MatrixXd& accelerations) {
        std::lock_guard<std::mutex> lock(mutex_);
        return controller_.compute_control(state, refs, accelerations);
    }
    State predict_state(const State& state, const Joint& tau, double duration) {
        std::lock_guard<std::mutex> lock(mutex_);
        return dynamics_.integrate(state, tau, duration);
    }
    std::tuple<State, MatA, MatB> linearize(const State& state, const Joint& tau, double duration) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto result = dynamics_.integrate_linearized(state, tau, duration);
        return {result.next, result.A, result.B};
    }
    Joint acceleration(const State& state, const Joint& tau) {
        std::lock_guard<std::mutex> lock(mutex_);
        return dynamics_.acceleration(state, tau);
    }
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        controller_.reset();
    }
    void set_torque_limits(const Joint& lower, const Joint& upper) {
        std::lock_guard<std::mutex> lock(mutex_);
        controller_.set_torque_limits(lower, upper);
    }
    Timing timing() {
        std::lock_guard<std::mutex> lock(mutex_);
        return controller_.timing();
    }
    double timestep() const { return controller_.timestep(); }
    int horizon() const { return controller_.horizon(); }
private:
    Dynamics dynamics_;
    Controller controller_;
    std::mutex mutex_;
};

PYBIND11_MODULE(rokae_mpc_v2, m) {
    m.doc() = "CPU MPC v2: analytical sensitivities and changing-Hessian QP warm start";
    py::class_<Timing>(m, "Timing")
        .def_readonly("reference_ms", &Timing::reference_ms)
        .def_readonly("dynamics_ms", &Timing::dynamics_ms)
        .def_readonly("matrices_ms", &Timing::matrices_ms)
        .def_readonly("qp_ms", &Timing::qp_ms)
        .def_readonly("total_ms", &Timing::total_ms)
        .def_readonly("qp_iterations", &Timing::qp_iterations)
        .def_readonly("qp_success", &Timing::qp_success);
    py::class_<MPCController>(m, "MPCController")
        .def(py::init<const std::string&, double, int, bool, double, bool>(),
             py::arg("urdf_path"), py::arg("timestep") = 0.001, py::arg("horizon") = 40,
             py::arg("gravity_compensated") = true, py::arg("integration_step") = 0.001,
             py::arg("warm_start") = true)
        .def_property_readonly("timestep", &MPCController::timestep)
        .def_property_readonly("horizon", &MPCController::horizon)
        .def_property_readonly("timing", &MPCController::timing)
        .def("compute_control", &MPCController::compute_control,
             py::arg("state"), py::arg("state_ref"), py::arg("ddq_ref"),
             py::call_guard<py::gil_scoped_release>())
        .def("predict_state", &MPCController::predict_state,
             py::arg("state"), py::arg("tau"), py::arg("duration"),
             py::call_guard<py::gil_scoped_release>())
        .def("linearize", &MPCController::linearize,
             py::arg("state"), py::arg("tau"), py::arg("duration"),
             py::call_guard<py::gil_scoped_release>())
        .def("acceleration", &MPCController::acceleration,
             py::arg("state"), py::arg("tau"), py::call_guard<py::gil_scoped_release>())
        .def("set_torque_limits", &MPCController::set_torque_limits,
             py::arg("lower"), py::arg("upper"), py::call_guard<py::gil_scoped_release>())
        .def("reset", &MPCController::reset, py::call_guard<py::gil_scoped_release>());
}
