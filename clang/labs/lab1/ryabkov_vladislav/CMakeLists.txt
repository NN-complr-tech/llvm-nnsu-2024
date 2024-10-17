add_llvm_library(ClassesRyabkovLabOne MODULE ClassesRyabkov.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(ClassesRyabkovLabOne PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()

set(CLANG_TEST_DEPS "ClassesRyabkovLabOne" ${CLANG_TEST_DEPS} PARENT_SCOPE)
