add_llvm_library(ShipitsinRenamePlugin MODULE ShipitsinRenamePlugin.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
    set(LLVM_LINK_COMPONENTS
            Support
    )
    clang_target_link_libraries(ShipitsinRenamePlugin PRIVATE
            clangAST
            clangBasic
            clangFrontend
    )
endif()

set(CLANG_TEST_DEPS "ShipitsinRenamePlugin" ${CLANG_TEST_DEPS} PARENT_SCOPE)