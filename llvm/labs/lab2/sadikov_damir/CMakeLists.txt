set(PluginName LoopLLVMPlugin_SadikovD)
add_compile_definitions(NAME=${PluginName})
if (NOT WIN32 AND NOT CYGWIN)
   add_llvm_pass_plugin(${PluginName} LoopLLVMPlugin_SadikovD.cpp)
   
   set(LLVM_TEST_DEPENDS ${PluginName} ${LLVM_TEST_DEPENDS} PARENT_SCOPE)
endif()