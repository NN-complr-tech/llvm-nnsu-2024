# Install script for directory: D:/llvm/llvm-nnsu-2024/mlir/lib/Conversion

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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/AffineToStandard/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/AMDGPUToROCDL/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/ArithCommon/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/ArithToLLVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/ArithToSPIRV/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/ArmNeon2dToIntr/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/AsyncToLLVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/BufferizationToMemRef/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/ComplexToLibm/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/ComplexToLLVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/ComplexToSPIRV/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/ComplexToStandard/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/ControlFlowToLLVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/ControlFlowToSPIRV/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/FuncToLLVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/FuncToSPIRV/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/GPUCommon/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/GPUToNVVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/GPUToROCDL/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/GPUToSPIRV/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/GPUToVulkan/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/IndexToLLVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/LinalgToLLVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/LinalgToStandard/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/LLVMCommon/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/MathToFuncs/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/MathToLibm/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/MathToLLVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/MathToSPIRV/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/MemRefToLLVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/MemRefToSPIRV/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/NVGPUToNVVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/NVVMToLLVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/OpenACCToSCF/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/OpenMPToLLVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/PDLToPDLInterp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/ReconcileUnrealizedCasts/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/SCFToControlFlow/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/SCFToGPU/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/SCFToOpenMP/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/SCFToSPIRV/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/ShapeToStandard/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/SPIRVToLLVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/TensorToLinalg/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/TensorToSPIRV/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/TosaToArith/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/TosaToLinalg/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/TosaToSCF/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/TosaToTensor/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/UBToLLVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/UBToSPIRV/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/VectorToArmSME/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/VectorToGPU/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/VectorToLLVM/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/VectorToSCF/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Conversion/VectorToSPIRV/cmake_install.cmake")
endif()

