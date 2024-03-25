add_llvm_library(InstrumentFunctionsPlugin MODULE instrumention_func.cpp PLUGIN_TOOL llvm)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(InstrumentFunctionsPlugin PUBLIC
    LLVMSupport
    LLVMCore
    LLVMipo
    LLVMPasses
    )
endif()

set(LLVM_TEST_DEPENDS "InstrumentFunctionsPlugin" ${LLVM_TEST_DEPENDS} PARENT_SCOPE)