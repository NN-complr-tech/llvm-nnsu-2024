# Install script for directory: D:/llvm/llvm-nnsu-2024/mlir/lib/ExecutionEngine

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "D:/llvm/llvm-nnsu-2024/mlir/out/install/x64-Debug")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRExecutionEngineUtilsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/ExecutionEngine/MLIRExecutionEngineUtils.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRExecutionEnginex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/ExecutionEngine/MLIRExecutionEngine.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRJitRunnerx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/ExecutionEngine/MLIRJitRunner.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmlir_float16_utilsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/ExecutionEngine/mlir_float16_utils.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmlir_float16_utilsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/bin/mlir_float16_utils.dll")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/ExecutionEngine/SparseTensor/cmake_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmlir_c_runner_utilsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/ExecutionEngine/mlir_c_runner_utils.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmlir_c_runner_utilsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/bin/mlir_c_runner_utils.dll")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmlir_runner_utilsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/ExecutionEngine/mlir_runner_utils.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmlir_runner_utilsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/bin/mlir_runner_utils.dll")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmlir_async_runtimex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/ExecutionEngine/mlir_async_runtime.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmlir_async_runtimex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/bin/mlir_async_runtime.dll")
endif()

