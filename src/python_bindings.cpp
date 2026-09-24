#include "NMPC_control.h"
#include "pinocchio_fun.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

// 这样pybind11::init就等价于py::init
namespace py = pybind11;

// 把动力学对象和控制器放在一起，保证 dynamics 的生命周期足够长。
class MPCController {
    pinocchioFun dynamics;
    ROKAE_NMPC controller;

public:
    explicit MPCController(const std::string& urdf_path, double timestep, int horizon)
        : dynamics(urdf_path), controller(dynamics, timestep, horizon) {}

    double timestep() const 
    { 
        return controller.timestep();
    }
    int horizon() const
    { 
        return controller.horizon(); 
    }

    // 这一段代码可以使得python代码也可以调用dynamics.compute_held_state()
    Eigen::VectorXd predict_state(const Eigen::VectorXd& state,
                                 const Eigen::VectorXd& tau, double duration)
    {
        if(state.size() != 12 || tau.size() != 6 || !state.allFinite() || !tau.allFinite()) {
            throw std::invalid_argument("Expected finite state (12,) and torque (6,).");
        }
        return dynamics.compute_held_state(state, tau, duration);
    }

    // 关键函数,state表示当前真实机器人状态，state_ref是整个预测时域上的参考状态
    Eigen::VectorXd compute_control(
        const Eigen::VectorXd& state,
        const Eigen::MatrixXd& state_ref,
        const Eigen::MatrixXd& ddq_ref)
    {
        const int N = horizon();
        // 尺寸检查， NumPy 的每一行对应一个预测时刻，避免错误输入导致 C++ 越界
        if(state.size() != 12 || state_ref.rows() != N+1 ||
           state_ref.cols() != 12 || ddq_ref.rows() != N || ddq_ref.cols() != 6) {
            throw std::invalid_argument(
                "Expected state (12,), state_ref (N+1, 12), ddq_ref (N, 6).");
        }
        // 检查三组输入都不能出现NaN/inf
        if(!state.allFinite() || !state_ref.allFinite() || !ddq_ref.allFinite()) {
            throw std::invalid_argument("Controller inputs must be finite.");
        }

        // 
        std::vector<Eigen::VectorXd> states(N+1), accelerations(N);

        for(int i = 0; i <= N; ++i) 
            states[i] = state_ref.row(i).transpose();
        for(int i = 0; i < N; ++i) 
            accelerations[i] = ddq_ref.row(i).transpose();

        Eigen::VectorXd tau = controller.compute_control(state, states, accelerations);
        // 控制器失败时会抛出异常；绑定层再检查输出，避免将无效力矩用于仿真。
        if(!controller.qp_success() || !tau.allFinite()) {
            throw std::runtime_error("MPC QP failed; tracking experiment stopped.");
        }
        return tau;
    }
};

// 创建一个python模块，名称为rokae_mpc，module为变量的名字
PYBIND11_MODULE(rokae_mpc, module) {
    // module.doc为Python的模块说明
    module.doc() = "ROKAE SR4 C++ MPC controller";
    // 把 C++ 的 MPCController 类注册进 rokae_mpc 模块，在 Python 里面也叫 MPCController
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
