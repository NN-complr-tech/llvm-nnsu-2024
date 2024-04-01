if (NOT WIN32 AND NOT CYGWIN)
   add_llvm_pass_plugin(LoopLLVMPlugin_SadikovD LoopLLVMPlugin_SadikovD.cpp)
   
   set(LLVM_TEST_DEPENDS LoopLLVMPlugin_SadikovD ${LLVM_TEST_DEPENDS} PARENT_SCOPE)
else()
   set(PluginName LoopLLVMPlugin_SadikovD)
   add_compile_definitions(NAME=${PluginName})
   
   string(TOUPPER ${PluginName} PluginNameUpper)
   set(LLVM_${PluginNameUpper}_LINK_INTO_TOOLS ON)
   
   add_llvm_pass_plugin(${PluginName} LoopLLVMPlugin_SadikovD.cpp)
   
   set(LLVM_TEST_DEPENDS ${PluginName} ${LLVM_TEST_DEPENDS} PARENT_SCOPE)
endif()