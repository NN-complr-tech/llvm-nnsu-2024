# Copyright 2024 Kulikov Artem
set(path "${CMAKE_CURRENT_SOURCE_DIR}")
cmake_path(GET path FILENAME ProjectId)
cmake_path(GET path PARENT_PATH Lab)
cmake_path(GET Lab FILENAME Lab)

set(ProjectId "${Lab}_${ProjectId}")
message( "--- " ${ProjectId} )


set(PluginName ${ProjectId})
add_compile_definitions(NAME=${PluginName})

string(TOUPPER ${PluginName} PluginNameUpper)
set(LLVM_${PluginNameUpper}_LINK_INTO_TOOLS ON)

if (NOT WIN32 AND NOT CYGWIN)
  file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.h)
  add_llvm_pass_plugin(${PluginName} 
  ${ALL_SOURCE_FILES}     
  DEPENDS
  intrinsics_gen
  BUILDTREE_ONLY)

  target_link_libraries(${PluginName}
    PUBLIC
    LLVMSupport
    LLVMCore
    LLVMipo
    LLVMPasses)

  set(LLVM_TEST_DEPENDS ${PluginName} ${LLVM_TEST_DEPENDS} PARENT_SCOPE)
endif()

