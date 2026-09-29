#pragma once
#include "mpc_controller_v3.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <functional>

namespace mpc_v3 {
namespace py=pybind11;
using BackendFactory=std::function<std::unique_ptr<DynamicsBackend>(
    const std::string&,const Config&,const std::string&,bool,const std::string&,const std::string&)>;

// CPU和FPGA模块导出相同接口；module_local允许在同一进程中加载两者做对照。
inline void bind_api(py::module_& m,const BackendFactory& factory,const std::string& default_backend) {
    py::class_<Config>(m,"Config",py::module_local())
        .def(py::init<>())
        .def_readwrite("horizon",&Config::horizon).def_readwrite("model_dt",&Config::model_dt)
        .def_readwrite("feedback_ratio",&Config::feedback_ratio)
        .def_readwrite("integration_step",&Config::integration_step)
        .def_readwrite("gravity_compensated",&Config::gravity_compensated)
        .def_readwrite("torque_limit",&Config::torque_limit).def_readwrite("torque_trust",&Config::torque_trust)
        .def_readwrite("position_trust",&Config::position_trust).def_readwrite("velocity_trust",&Config::velocity_trust)
        .def_readwrite("max_position_defect",&Config::max_position_defect)
        .def_readwrite("max_velocity_defect",&Config::max_velocity_defect)
        .def_readwrite("max_measurement_age",&Config::max_measurement_age)
        .def_readwrite("enforce_feedback_budget",&Config::enforce_feedback_budget)
        .def_readwrite("qp_iterations",&Config::qp_iterations)
        .def_property_readonly("feedback_dt",&Config::feedback_dt);
    py::class_<BackendStats>(m,"BackendStats",py::module_local())
        .def_readonly("batches",&BackendStats::batches).def_readonly("samples",&BackendStats::samples)
        .def_readonly("transfer_and_wait_ms",&BackendStats::transfer_and_wait_ms);
    py::class_<PreparationInfo>(m,"PreparationInfo",py::module_local())
        .def_readonly("block",&PreparationInfo::block).def_readonly("model_ms",&PreparationInfo::model_ms)
        .def_readonly("matrices_and_qp_ms",&PreparationInfo::matrices_and_qp_ms)
        .def_readonly("total_ms",&PreparationInfo::total_ms)
        .def_readonly("max_position_defect",&PreparationInfo::max_position_defect)
        .def_readonly("max_velocity_defect",&PreparationInfo::max_velocity_defect)
        .def_readonly("backend",&PreparationInfo::backend);
    py::class_<FeedbackResult>(m,"FeedbackResult",py::module_local())
        .def_readonly("torque",&FeedbackResult::torque).def_readonly("states",&FeedbackResult::states)
        .def_readonly("controls",&FeedbackResult::controls).def_readonly("state_times",&FeedbackResult::state_times)
        .def_readonly("tick",&FeedbackResult::tick).def_readonly("block",&FeedbackResult::block)
        .def_readonly("phase",&FeedbackResult::phase).def_readonly("solve_ms",&FeedbackResult::solve_ms)
        .def_readonly("total_ms",&FeedbackResult::total_ms).def_readonly("qp_iterations",&FeedbackResult::qp_iterations)
        .def_readonly("deadline_missed",&FeedbackResult::deadline_missed);
    py::class_<Controller>(m,"MPCController",py::module_local())
        .def(py::init([factory](const std::string& urdf,Config c,const std::string& backend,
                                bool allow_hardware,const std::string& h2c,const std::string& c2h) {
            c.validate();
            return std::make_unique<Controller>(c,factory(urdf,c,backend,allow_hardware,h2c,c2h));
        }),py::arg("urdf_path"),py::arg("config"),py::arg("backend")=default_backend,
           py::arg("allow_hardware")=false,py::arg("h2c")="/dev/xdma0_h2c_0",py::arg("c2h")="/dev/xdma0_c2h_0")
        .def("prepare",&Controller::prepare,py::arg("block"),py::arg("anchor_state"),
             py::arg("state_ref"),py::arg("ddq_ref"),py::call_guard<py::gil_scoped_release>())
        .def("feedback",&Controller::feedback,py::arg("state"),py::arg("tick"),
             py::arg("measurement_age_s")=0,py::call_guard<py::gil_scoped_release>())
        .def("validate_backend",&Controller::validate_backend,py::arg("states"),py::arg("torques"),
             py::arg("absolute_tolerance")=0.05,py::arg("relative_tolerance")=0.002,
             py::call_guard<py::gil_scoped_release>())
        .def("has_model",&Controller::has_model,py::call_guard<py::gil_scoped_release>())
        .def("reset",&Controller::reset,py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("config",&Controller::config)
        .def_property_readonly("backend_name",&Controller::backend_name);
}
} // namespace mpc_v3
