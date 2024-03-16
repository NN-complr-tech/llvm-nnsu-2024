add_llvm_library(depWarningPlugin MODULE deprecatedWarning.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
      Support
  )
  clang_target_link_libraries(depWarningPlugin PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()

set(CLANG_TEST_DEPS "depWarningPlugin" ${CLANG_TEST_DEPS} PARENT_SCOPE)
