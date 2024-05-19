cmake_minimum_required(VERSION 3.22)
project(ProjectName)

# новые политики для CMP0114 и CMP0116
cmake_policy(SET CMP0114 NEW)
cmake_policy(SET CMP0116 NEW)

# необходимые модули
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)  # модуль AddLLVM.cmake, который предоставляет add_llvm_* функции

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

set(PluginName AkopyanDiv)

file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.h)
add_llvm_pass_plugin(${PluginName}
        ${ALL_SOURCE_FILES}
        DEPENDS
        intrinsics_gen
        MLIRBuiltinLocationAttributesIncGen
        BUILDTREE_ONLY
        )

set(MLIR_TEST_DEPENDS ${PluginName} ${MLIR_TEST_DEPENDS} PARENT_SCOPE)