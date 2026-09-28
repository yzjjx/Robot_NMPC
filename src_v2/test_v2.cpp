#include "NMPC_control.h"
#include <iostream>
#include <limits>

using namespace mpc_v2;
namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class Function> void must_throw(Function function) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    check(rejected, "Invalid input was not rejected");
}

void derivative_check(Dynamics& d) {
    State x;
    for (int j = 0; j < DOF; ++j) {
        x(j) = 0.2*std::sin(j+1.0);
        x(j+DOF) = 0.01*std::cos(j+1.0);
    }
    const Joint u = d.inverse_dynamics(x, Joint::Zero());
    for (double dt : {0.0, 0.001, 0.01, 0.1}) {
        const auto exact = d.integrate_linearized(x, u, dt);
        check((exact.next-d.integrate(x, u, dt)).norm() < 1e-10,
              "Linearized and plain integration disagree");
        MatA fd_a;
        MatB fd_b;
        for (int j = 0; j < NX; ++j) {
            State plus = x, minus = x;
            plus(j) += 1e-6;
            minus(j) -= 1e-6;
            fd_a.col(j) = (d.integrate(plus,u,dt)-d.integrate(minus,u,dt))/(2e-6);
        }
        for (int j = 0; j < DOF; ++j) {
            Joint plus = u, minus = u;
            plus(j) += 1e-5;
            minus(j) -= 1e-5;
            fd_b.col(j) = (d.integrate(x,plus,dt)-d.integrate(x,minus,dt))/(2e-5);
        }
        const double ea = ((exact.A-fd_a).array().abs()/(1+fd_a.array().abs())).maxCoeff();
        const double eb = ((exact.B-fd_b).array().abs()/(1+fd_b.array().abs())).maxCoeff();
        std::cout << "duration=" << dt << " A_error=" << ea << " B_error=" << eb << "\n";
        check(ea < 2e-4 && eb < 2e-4, "Analytical derivative check failed");
    }
    must_throw([&]{ d.integrate(x,u,-0.01); });
    x(0) = std::numeric_limits<double>::quiet_NaN();
    must_throw([&]{ d.integrate_linearized(x,u,0.01); });
}

void condensing_check() {
    const int n = 3;
    std::vector<MatA> a(n, MatA::Identity());
    std::vector<MatB> b(n, MatB::Zero());
    for (int i = 0; i < n; ++i) {
        a[i].topRightCorner<DOF,DOF>() = 0.01*Eigen::Matrix<double,DOF,DOF>::Identity();
        b[i].bottomRows<DOF>() = (i+1.0)*0.01*Eigen::Matrix<double,DOF,DOF>::Identity();
    }
    MPCMatrices matrices(n);
    matrices.update(a,b);
    const Eigen::VectorXd du = Eigen::VectorXd::LinSpaced(n*DOF,-0.1,0.2);
    const Eigen::VectorXd condensed = matrices.Gamma*du;
    State dx = State::Zero();
    for (int i = 0; i < n; ++i) {
        dx = (a[i]*dx + b[i]*du.segment<DOF>(i*DOF)).eval();
        check((dx-condensed.segment<NX>(i*NX)).norm() < 1e-12, "Gamma recursion failed");
    }
}

void qp_check() {
    const int n = 2;
    MPCMatrices m(n);
    Prediction warm(n, true), cold(n, false);
    const Eigen::VectorXd nominal = Eigen::VectorXd::Zero(n*DOF);
    const Joint lower = Joint::Constant(-0.15), upper = Joint::Constant(0.15);
    for (int k = 0; k < 8; ++k) {
        // 每次改变 Gamma，确保 H 确实改变；同时交替激活上下界。
        for (int r = 0; r < m.Gamma.rows(); ++r)
            for (int c = 0; c < m.Gamma.cols(); ++c)
                m.Gamma(r,c) = 0.01*std::sin(0.4*r+0.7*c+0.2*k);
        m.weighted_Gamma = m.Gamma;
        for (int r = 0; r < m.Gamma.rows(); ++r)
            m.weighted_Gamma.row(r) *= m.state_weights(r);
        const Eigen::VectorXd error = Eigen::VectorXd::Constant(n*NX, k%2 ? 0.2 : -0.2);
        warm.prepare(m,error,nominal,nominal,lower,upper);
        cold.prepare(m,error,nominal,nominal,lower,upper);
        const Eigen::VectorXd uw = warm.solve();
        const Eigen::VectorXd uc = cold.solve();
        check((uw-uc).lpNorm<Eigen::Infinity>() < 1e-5, "Changed-H warm/cold solutions differ");
        check(uw.minCoeff() >= -0.15001 && uw.maxCoeff() <= 0.15001, "Torque bound failure");
    }
}

void controller_check(Dynamics& d) {
    Controller warm(d,0.01,3,true), cold(d,0.01,3,false);
    Eigen::MatrixXd refs = Eigen::MatrixXd::Zero(4,NX);
    Eigen::MatrixXd ddq = Eigen::MatrixXd::Zero(3,DOF);
    State x = State::Zero();
    for (int k = 0; k < 6; ++k) {
        x(0) = 0.005*std::sin(0.3*k);
        const Joint uw = warm.compute_control(x,refs,ddq);
        const Joint uc = cold.compute_control(x,refs,ddq);
        check((uw-uc).norm() < 1e-4, "Controller warm/cold mismatch");
        check(warm.timing().qp_success, "Missing success status");
    }
    must_throw([&]{ Controller invalid(d,0.01,0); });
    must_throw([&]{ warm.set_torque_limits(Joint::Ones(),Joint::Zero()); });
    Eigen::MatrixXd wrong = Eigen::MatrixXd::Zero(2,NX);
    must_throw([&]{ warm.compute_control(x,wrong,ddq); });
    check(!warm.timing().qp_success, "Failure did not clear success flag");
    warm.compute_control(x,refs,ddq); // 失败后应能重新冷启动。
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Usage: test_mpc_v2 URDF");
        for (bool compensated : {true, false}) {
            Dynamics dynamics(argv[1],compensated);
            derivative_check(dynamics);
        }
        condensing_check();
        qp_check();
        Dynamics dynamics(argv[1]);
        controller_check(dynamics);
        std::cout << "All numerical checks passed. Not a hardware safety validation.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
