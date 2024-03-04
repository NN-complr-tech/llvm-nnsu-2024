# set(LLVM_LINK_COMPONENTS Support)

# add_executable(clangDepMatcher deprecatedMatcher.cpp)

# target_link_libraries(clangDepMatcher clangTooling)

# add_clang_library(clangDepMatcher
#   deprecatedMatcher.cpp

#   LINK_LIBS
#   clangAST
#   clangASTMatchers
#   clangBasic
#   clangFrontend
#   clangSerialization
#   clangTooling
#   )

# If we don't need RTTI or EH, there's no reason to export anything
# from the plugin.
add_llvm_library(clangDepMatcher MODULE deprecatedMatcher.cpp PLUGIN_TOOL clang)


# tmp
# add_clang_library(clangDepMatcher
    
#     deprecatedMatcher.cpp
    
#     LINK_LIBS
#     clangAST
#     clangASTMatchers
#     clangBasic
#     clangFrontend
#     clangSerialization
#     clangTooling
#   )

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
      Support
  )
  clang_target_link_libraries(clangDepMatcher PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()

set(CLANG_TEST_DEPS "clangDepMatcher" ${CLANG_TEST_DEPS} PARENT_SCOPE)
