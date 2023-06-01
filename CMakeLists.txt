cmake_minimum_required(VERSION 2.8) # Cmake 最低版本要求

#
## 生成ORBSLAM2工程
#

project(ORB_SLAM2)   # 工程名 PROJECT_NAME = ORB_SLAM2

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# 指定 C语言的编译选项
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
# 指定 C++语言的编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# 引入CheckCXXCompilerFlag函数
include(CheckCXXCompilerFlag)
# 调用CheckCXXCompilerFlag函数检测编译器是否支持-std=c++11标志
# 是否支持的结果存放在COMPILER_SUPPORTS_CXX11中，（true: 支持，False: 不支持）
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX11) # 是否支持C++11
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)        # 把宏定义DCOMPILEDWITHC11添加到程序中
   message(STATUS "Using flag -std=c++11.")  # 输出提示信息，说明编译器正在用C++11进行编译
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.0 QUIET)      # 先找 OpenCV 3.0版本
if(NOT OpenCV_FOUND)                # 没找到
   find_package(OpenCV 2.4.3 QUIET) # 找 OpenCV 2.4.3版本
   if(NOT OpenCV_FOUND)             # 还没找到
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")   # 输出提示信息
   endif()
endif()
# QUIET：无提示信息
# REQUIRED：有提示信息
find_package(Eigen3 3.1.0 REQUIRED) # 找 Eigen3 3.1.0版本
find_package(Pangolin REQUIRED)     # 找 Pangolin 库

# 添加头文件
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# 库文件
add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
)

# 链接依赖库
target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

#
## 生成可执行的二进制文件
#

# ------------------------- RGBD -------------------------------------
# 设置可执行文件的输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Bin/RGB-D)
# 将源文件（后者）编译成可执行文件（前者）
add_executable(rgbd_tum Examples/RGB-D/rgbd_tum.cc)
# 将 ORBSLAM2工程链接到可执行文件
target_link_libraries(rgbd_tum ${PROJECT_NAME})


# ------------------------- 双目 -------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Bin/Stereo)
# 双目——kitti数据集
add_executable(stereo_kitti Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})
# 双目——euroc数据集
add_executable(stereo_euroc Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})


# ------------------------- 单目 -------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Bin/Monocular)
# 单目——tum数据集
add_executable(mono_tum Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})
# 单目——kitti数据集
add_executable(mono_kitti Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})
# 单目——euroc数据集
add_executable(mono_euroc Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

