# Install script for directory: D:/llvm/llvm-nnsu-2024/mlir/lib/Interfaces

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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCallInterfacesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRCallInterfaces.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCastInterfacesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRCastInterfaces.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRControlFlowInterfacesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRControlFlowInterfaces.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCopyOpInterfacex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRCopyOpInterface.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRDataLayoutInterfacesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRDataLayoutInterfaces.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRDerivedAttributeOpInterfacex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRDerivedAttributeOpInterface.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRDestinationStyleOpInterfacex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRDestinationStyleOpInterface.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRInferIntRangeInterfacex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRInferIntRangeInterface.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRInferTypeOpInterfacex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRInferTypeOpInterface.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRLoopLikeInterfacex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRLoopLikeInterface.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRMemorySlotInterfacesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRMemorySlotInterfaces.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRParallelCombiningOpInterfacex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRParallelCombiningOpInterface.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRRuntimeVerifiableOpInterfacex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRRuntimeVerifiableOpInterface.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRShapedOpInterfacesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRShapedOpInterfaces.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRSideEffectInterfacesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRSideEffectInterfaces.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRTilingInterfacex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRTilingInterface.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRVectorInterfacesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRVectorInterfaces.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRViewLikeInterfacex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRViewLikeInterface.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRValueBoundsOpInterfacex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/MLIRValueBoundsOpInterface.lib")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/Interfaces/Utils/cmake_install.cmake")
endif()

