cmake_minimum_required(VERSION 3.18)
project(Robot_NMPC_v2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 独立构建，完全不修改上一级 CMakeLists.txt 或旧 rokae_mpc 模块。
list(APPEND CMAKE_PREFIX_PATH "/opt/openrobots")
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(qpOASES REQUIRED)
find_package(Threads REQUIRED)

add_library(mpc_core_v2 STATIC
    pinocchio_fun.cpp MPC_Matrices.cpp Prediction.cpp NMPC_control.cpp)
set_target_properties(mpc_core_v2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mpc_core_v2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    ${qpOASES_INCLUDE_DIR} ${qpOASES_INCLUDE_DIRS})
target_link_libraries(mpc_core_v2 PUBLIC Eigen3::Eigen pinocchio::pinocchio Threads::Threads)
if(TARGET qpOASES::qpOASES)
    target_link_libraries(mpc_core_v2 PUBLIC qpOASES::qpOASES)
elseif(TARGET qpOASES)
    target_link_libraries(mpc_core_v2 PUBLIC qpOASES)
elseif(qpOASES_LIBRARIES)
    target_link_libraries(mpc_core_v2 PUBLIC ${qpOASES_LIBRARIES})
elseif(qpOASES_LIBRARY)
    target_link_libraries(mpc_core_v2 PUBLIC ${qpOASES_LIBRARY})
else()
    message(FATAL_ERROR "qpOASES was found but no library target/path is available")
endif()

add_executable(mpc_benchmark main.cpp)
target_link_libraries(mpc_benchmark PRIVATE mpc_core_v2)

option(BUILD_PYTHON_BINDINGS "Build independent rokae_mpc_v2 module" ON)
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(PYBIND11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    endif()
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(rokae_mpc_v2 python_bindings.cpp)
    target_link_libraries(rokae_mpc_v2 PRIVATE mpc_core_v2)
    # 模块留在构建目录，使用 PYTHONPATH 访问，不覆盖 python_code。
    set_target_properties(rokae_mpc_v2 PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test_mpc_v2 test_v2.cpp)
    target_link_libraries(test_mpc_v2 PRIVATE mpc_core_v2)
    set(MPC_TEST_URDF "${CMAKE_CURRENT_SOURCE_DIR}/../urdf/ROKAE_SR4.urdf"
        CACHE FILEPATH "URDF used by the numerical checks")
    add_test(NAME numerical_checks COMMAND test_mpc_v2 "${MPC_TEST_URDF}")
endif()
