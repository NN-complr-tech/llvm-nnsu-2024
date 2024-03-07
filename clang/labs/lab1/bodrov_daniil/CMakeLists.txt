add_llvm_library(DeprecatedWarnPlugin MODULE DeprecatedWarn.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(DeprecatedWarnPlugin PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()

set(CLANG_TEST_DEPS "DeprecatedWarnPlugin" ${CLANG_TEST_DEPS} PARENT_SCOPE)