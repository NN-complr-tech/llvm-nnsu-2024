add_llvm_library(ShmelevPrintNamesPlugin MODULE ShmelevPrintNamesPlugin.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(ShmelevPrintNamesPlugin PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()

set(CLANG_TEST_DEPS "ShmelevPrintNamesPlugin" ${CLANG_TEST_DEPS} PARENT_SCOPE)
