add_llvm_library(InstrumentFunctionsPlugin MODULE instrumention_func.cpp PLUGIN_TOOL llvm)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  target_link_libraries(InstrumentFunctionsPlugin PRIVATE
    LLVMSupport
    LLVMCore
    LLVMipo
    LLVMPasses
    )
endif()

set(LLVM_TEST_DEPENDS "InstrumentFunctionsPlugin" ${LLVM_TEST_DEPENDS} PARENT_SCOPE)