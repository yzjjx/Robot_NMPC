cmake_minimum_required(VERSION 3.18)
project(Robot_NMPC)

set(CMAKE_CXX_STANDARD 17)

# 添加这一句，才能找到pinocchio
list(APPEND CMAKE_PREFIX_PATH "/opt/openrobots")

find_package(Eigen3 REQUIRED)
find_package(qpOASES REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Threads REQUIRED)

add_library(mpc_core STATIC
    src/MPC_Matrices.cpp
    src/NMPC_control.cpp
    src/pinocchio_fun.cpp
    src/Prediction.cpp
)
set_target_properties(mpc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
    mpc_core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # 需要添加Pinocchio的头文件
    /opt/openrobots/include
    # 添加qp
    ${qpOASES_INCLUDE_DIR}
)

# 链接
target_link_libraries(
    mpc_core
    PUBLIC
    Eigen3::Eigen
    pinocchio::pinocchio
    ${qpOASES_LIBRARY}
)

add_executable(Robot_NMPC 
    src/main.cpp)
target_link_libraries(Robot_NMPC PRIVATE mpc_core)

# # 珞石机器人实机控制程序。
# set(XCORE_SDK "/home/yzj_robot/xCoreSDK_cpp-v0.7.2" CACHE PATH
#     "Path to the xCoreSDK root directory")
# set(XCORE_LIB_DIR "${XCORE_SDK}/lib/Linux/x86_64")

# find_library(ROKAE_LIB
#     NAMES xCoreSDK
#     PATHS "${XCORE_LIB_DIR}"
#     NO_DEFAULT_PATH
#     REQUIRED
# )
# find_library(XMATE_MODEL_LIB
#     NAMES xMateModel
#     PATHS "${XCORE_LIB_DIR}"
#     NO_DEFAULT_PATH
#     REQUIRED
# )

# add_executable(MPC_con Rokae_code/MPC_con.cpp)
# target_include_directories(MPC_con PRIVATE "${XCORE_SDK}/include")
# target_link_libraries(MPC_con PRIVATE
#     mpc_core
#     ${ROKAE_LIB}
#     ${XMATE_MODEL_LIB}
#     Threads::Threads
# )

# 使用同一个控制器实现生成 Python 模块，不在 Python 中重写 MPC。
option(BUILD_PYTHON_BINDINGS "Build the pybind11 controller module" ON)
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(rokae_mpc src/python_bindings.cpp)
    target_link_libraries(rokae_mpc PRIVATE mpc_core)
    # 与仿真脚本放在一起，可以直接 import rokae_mpc。
    set_target_properties(rokae_mpc PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python_code"
    )
endif()
