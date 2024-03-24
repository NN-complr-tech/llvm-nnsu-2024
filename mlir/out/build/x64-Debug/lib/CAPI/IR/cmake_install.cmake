# Install script for directory: D:/llvm/llvm-nnsu-2024/mlir/lib/CAPI/IR

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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPIIRx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/IR/MLIRCAPIIR.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPIIRx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPIIR" TYPE FILE FILES
    "AffineExpr.cpp.obj"
    "AffineMap.cpp.obj"
    "BuiltinAttributes.cpp.obj"
    "BuiltinTypes.cpp.obj"
    "Diagnostics.cpp.obj"
    "DialectHandle.cpp.obj"
    "IntegerSet.cpp.obj"
    "IR.cpp.obj"
    "Pass.cpp.obj"
    "Support.cpp.obj"
    FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/IR/CMakeFiles/obj.MLIRCAPIIR.dir/./")
endif()

