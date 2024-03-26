add_llvm_library(opCounter MODULE test.cpp PLUGIN_TOOL opt)

# if(WIN32 OR CYGWIN)
#   set(LLVM_LINK_COMPONENTS
#       Support
#       Core
#   )
# endif()

# add_library(test.cpp)

# llvm_map_components_to_libnames(llvmLibs Passes Support Core)