
set(PluginName AtikinLoopPass)
add_compile_definitions(NAME=${PluginName})

string(TOUPPER ${PluginName} PluginNameUpper)
set(LLVM_${PluginNameUpper}_LINK_INTO_TOOLS ON)

file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.h)
add_llvm_pass_plugin(${PluginName} 
${ALL_SOURCE_FILES}     
DEPENDS
intrinsics_gen
BUILDTREE_ONLY)

target_link_libraries(${PluginName}
  PUBLIC
  LLVMSupport
  LLVMCore
  LLVMipo
  LLVMPasses)

set(LLVM_TEST_DEPENDS ${PluginName} ${LLVM_TEST_DEPENDS} PARENT_SCOPE)
