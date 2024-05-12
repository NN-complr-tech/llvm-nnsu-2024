set(PluginName ZakharovFuncCallCnt)

file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.h)
add_llvm_pass_plugin(${PluginName}
        ${ALL_SOURCE_FILES}
        DEPENDS
        intrinsics_gen
        BUILDTREE_ONLY
        )

target_include_directories(${PluginName} PUBLIC ${PATH_TO_TOOLS_MLIR})

set(LLVM_TEST_DEPENDS ${PluginName} ${LLVM_TEST_DEPENDS} PARENT_SCOPE)
