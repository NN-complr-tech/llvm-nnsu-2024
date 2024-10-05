add_llvm_library(MirzakhmedovLabaOnePlugin MODULE mirzakhmedovClassesPrint.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(MirzakhmedovLabaOnePlugin PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()

set(CLANG_TEST_DEPS "MirzakhmedovLabaOnePlugin" ${CLANG_TEST_DEPS} PARENT_SCOPE)
