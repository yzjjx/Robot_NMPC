cmake_minimum_required(VERSION 3.10)
project(Robot_NMPC)

set(CMAKE_CXX_STANDARD 17)

# 添加这一句，才能找到pinocchio
list(APPEND CMAKE_PREFIX_PATH "/opt/openrobots")

find_package(Eigen3 REQUIRED)
find_package(qpOASES REQUIRED)
find_package(pinocchio REQUIRED)

add_executable(Robot_NMPC
    src/main.cpp
    src/MPC_Matrices.cpp
    src/NMPC_control.cpp
    src/pinocchio_fun.cpp
    src/Prediction.cpp
)

target_include_directories(
    Robot_NMPC
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # 需要添加Pinocchio的头文件
    /opt/openrobots/include
    # 添加qp
    ${qpOASES_INCLUDE_DIR}
)


# 链接
target_link_libraries(
    Robot_NMPC
    PRIVATE
    Eigen3::Eigen
    pinocchio::pinocchio
    ${qpOASES_LIBRARY}
)