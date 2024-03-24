# This file allows users to call find_package(LLD) and pick up our targets.



set(LLVM_VERSION 17.0.6)
find_package(LLVM ${LLVM_VERSION} EXACT REQUIRED CONFIG
             HINTS "C:/Program Files (x86)/LLVM/lib/cmake/llvm")

set(LLD_EXPORTED_TARGETS "lldCommon;lld;lldCOFF;lldELF;lldMachO;lldMinGW;lldWasm")
set(LLD_CMAKE_DIR "D:/llvm/llvm-nnsu-2024/lld/out/build/x64-Debug/lib/cmake/lld")
set(LLD_INCLUDE_DIRS "D:/llvm/llvm-nnsu-2024/lld/include;D:/llvm/llvm-nnsu-2024/lld/out/build/x64-Debug/include")

# Provide all our library targets to users.
include("D:/llvm/llvm-nnsu-2024/lld/out/build/x64-Debug/lib/cmake/lld/LLDTargets.cmake")
