add_llvm_library(RenamedIdPlugin MODULE RenameIdentificator.cpp PLUGIN_TOOL clang)

set(LLVM_LINK_COMPONENTS
  Support
)
clang_target_link_libraries(RenamedIdPlugin PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangTooling
  clangTransformer
)

set(CLANG_TEST_DEPS "RenamedIdPlugin" ${CLANG_TEST_DEPS} PARENT_SCOPE)