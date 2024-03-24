add_llvm_library(HelloPlugin MODULE hello.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(HelloPlugin PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()

set(CLANG_TEST_DEPS "HelloPlugin" ${CLANG_TEST_DEPS} PARENT_SCOPE)