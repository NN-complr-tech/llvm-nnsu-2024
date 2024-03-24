# Install script for directory: D:/llvm/llvm-nnsu-2024/mlir/lib/CAPI/Dialect

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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPIArithx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPIArith.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPIArithx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPIArith" TYPE FILE FILES "Arith.cpp.obj" FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPIArith.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPIAsyncx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPIAsync.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPIAsyncx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPIAsync" TYPE FILE FILES
    "Async.cpp.obj"
    "AsyncPasses.cpp.obj"
    FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPIAsync.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPIControlFlowx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPIControlFlow.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPIControlFlowx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPIControlFlow" TYPE FILE FILES "ControlFlow.cpp.obj" FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPIControlFlow.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPIMathx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPIMath.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPIMathx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPIMath" TYPE FILE FILES "Math.cpp.obj" FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPIMath.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPIMemRefx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPIMemRef.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPIMemRefx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPIMemRef" TYPE FILE FILES "MemRef.cpp.obj" FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPIMemRef.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPIGPUx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPIGPU.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPIGPUx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPIGPU" TYPE FILE FILES
    "GPU.cpp.obj"
    "GPUPasses.cpp.obj"
    FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPIGPU.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPILLVMx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPILLVM.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPILLVMx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPILLVM" TYPE FILE FILES "LLVM.cpp.obj" FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPILLVM.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPILinalgx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPILinalg.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPILinalgx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPILinalg" TYPE FILE FILES
    "Linalg.cpp.obj"
    "LinalgPasses.cpp.obj"
    FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPILinalg.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPIMLProgramx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPIMLProgram.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPIMLProgramx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPIMLProgram" TYPE FILE FILES "MLProgram.cpp.obj" FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPIMLProgram.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPISCFx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPISCF.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPISCFx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPISCF" TYPE FILE FILES "SCF.cpp.obj" FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPISCF.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPIShapex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPIShape.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPIShapex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPIShape" TYPE FILE FILES "Shape.cpp.obj" FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPIShape.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPISparseTensorx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPISparseTensor.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPISparseTensorx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPISparseTensor" TYPE FILE FILES
    "SparseTensor.cpp.obj"
    "SparseTensorPasses.cpp.obj"
    FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPISparseTensor.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPIFuncx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPIFunc.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPIFuncx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPIFunc" TYPE FILE FILES "Func.cpp.obj" FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPIFunc.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPITensorx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPITensor.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPITensorx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPITensor" TYPE FILE FILES "Tensor.cpp.obj" FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPITensor.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPITransformDialectx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPITransformDialect.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPITransformDialectx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPITransformDialect" TYPE FILE FILES "Transform.cpp.obj" FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPITransformDialect.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPIQuantx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPIQuant.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPIQuantx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPIQuant" TYPE FILE FILES "Quant.cpp.obj" FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPIQuant.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPIPDLx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPIPDL.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPIPDLx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPIPDL" TYPE FILE FILES "PDL.cpp.obj" FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPIPDL.dir/./")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xMLIRCAPIVectorx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/MLIRCAPIVector.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xobj.MLIRCAPIVectorx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Debug/obj.MLIRCAPIVector" TYPE FILE FILES "Vector.cpp.obj" FILES_FROM_DIR "D:/llvm/llvm-nnsu-2024/mlir/out/build/x64-Debug/lib/CAPI/Dialect/CMakeFiles/obj.MLIRCAPIVector.dir/./")
endif()

