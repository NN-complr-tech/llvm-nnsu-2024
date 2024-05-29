add_llvm_library(BozinAlwaysInlinePlugin MODULE AlwaysInline.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(BozinAlwaysInlinePlugin PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()

set(CLANG_TEST_DEPS "BozinAlwaysInlinePlugin" ${CLANG_TEST_DEPS} PARENT_SCOPE)