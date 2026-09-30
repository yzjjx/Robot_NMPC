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
    MPCController(
        const std::string& urdf_path,
        double timestep,
        int horizon,
        bool gravity_compensated,
        double integration_step,
        bool warm_start
    )
        : dynamics(urdf_path, gravity_compensated, integration_step),
          controller(dynamics, timestep, horizon, warm_start)
    {
    }

    Joint compute_control(
        const State& state,
        const Eigen::MatrixXd& state_ref,
        const Eigen::MatrixXd& ddq_ref
    )
    {
        std::lock_guard<std::mutex> lock(controller_mutex);
        return controller.compute_control(state, state_ref, ddq_ref);
    }

    State predict_state(
        const State& state,
        const Joint& tau,
        double duration
    )
    {
        std::lock_guard<std::mutex> lock(controller_mutex);
        return dynamics.integrate(state, tau, duration);
    }

    std::tuple<State, MatA, MatB> linearize(
        const State& state,
        const Joint& tau,
        double duration
    )
    {
        std::lock_guard<std::mutex> lock(controller_mutex);
        const auto result = dynamics.integrate_linearized(state, tau, duration);
        return {result.next, result.A, result.B};
    }

    Joint acceleration(
        const State& state,
        const Joint& tau
    )
    {
        std::lock_guard<std::mutex> lock(controller_mutex);
        return dynamics.acceleration(state, tau);
    }

    void reset()
    {
        std::lock_guard<std::mutex> lock(controller_mutex);
        controller.reset();
    }

    void set_torque_limits(
        const Joint& tau_lower,
        const Joint& tau_upper
    )
    {
        std::lock_guard<std::mutex> lock(controller_mutex);
        controller.set_torque_limits(tau_lower, tau_upper);
    }

    Timing timing()
    {
        std::lock_guard<std::mutex> lock(controller_mutex);
        return controller.timing();
    }

    double timestep() const
    {
        return controller.timestep();
    }

    int horizon() const
    {
        return controller.horizon();
    }

private:
    Dynamics dynamics;
    Controller controller;
    std::mutex controller_mutex;
};

PYBIND11_MODULE(
    rokae_mpc_v2,
    module
)
{
    module.doc() = "CPU MPC v2: analytical sensitivities and changing-Hessian QP warm start";
    py::class_<Timing>(module, "Timing")
        .def_readonly("reference_ms", &Timing::reference_ms)
        .def_readonly("dynamics_ms", &Timing::dynamics_ms)
        .def_readonly("matrices_ms", &Timing::matrices_ms)
        .def_readonly("qp_ms", &Timing::qp_ms)
        .def_readonly("total_ms", &Timing::total_ms)
        .def_readonly("qp_iterations", &Timing::qp_iterations)
        .def_readonly("qp_success", &Timing::qp_success);
    py::class_<MPCController>(module, "MPCController")
        .def(
            py::init<const std::string&, double, int, bool, double, bool>(),
            py::arg("urdf_path"),
            py::arg("timestep") = 0.001,
            py::arg("horizon") = 40,
            py::arg("gravity_compensated") = true,
            py::arg("integration_step") = 0.001,
            py::arg("warm_start") = true
        )
        .def_property_readonly("timestep", &MPCController::timestep)
        .def_property_readonly("horizon", &MPCController::horizon)
        .def_property_readonly("timing", &MPCController::timing)
        .def(
            "compute_control",
            &MPCController::compute_control,
            py::arg("state"),
            py::arg("state_ref"),
            py::arg("ddq_ref"),
            py::call_guard<py::gil_scoped_release>()
        )
        .def(
            "predict_state",
            &MPCController::predict_state,
            py::arg("state"),
            py::arg("tau"),
            py::arg("duration"),
            py::call_guard<py::gil_scoped_release>()
        )
        .def(
            "linearize",
            &MPCController::linearize,
            py::arg("state"),
            py::arg("tau"),
            py::arg("duration"),
            py::call_guard<py::gil_scoped_release>()
        )
        .def(
            "acceleration",
            &MPCController::acceleration,
            py::arg("state"),
            py::arg("tau"),
            py::call_guard<py::gil_scoped_release>()
        )
        .def(
            "set_torque_limits",
            &MPCController::set_torque_limits,
            py::arg("lower"),
            py::arg("upper"),
            py::call_guard<py::gil_scoped_release>()
        )
        .def("reset", &MPCController::reset, py::call_guard<py::gil_scoped_release>());
}
