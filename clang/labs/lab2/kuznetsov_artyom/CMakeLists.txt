add_llvm_library(InstrumentFunctionsPlugin MODULE instrumention_func.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(InstrumentFunctionsPlugin PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()

set(CLANG_TEST_DEPS "InstrumentFunctionsPlugin" ${CLANG_TEST_DEPS} PARENT_SCOPE)