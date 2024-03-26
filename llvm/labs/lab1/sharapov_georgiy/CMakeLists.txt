set(LLVM_LOOPFRAMER_LINK_INTO_TOOLS ON)

add_llvm_pass_plugin(LoopFramer LoopFramer.cpp)

if(WIN32 OR CYGWIN)
    set(LLVM_LINK_COMPONENTS
        Support
    )
    clang_target_link_libraries(DepWarningPlugin PRIVATE
        Analysis      
        Core          
        Support       
        TransformUtils
    )
endif()

set(LLVM_TEST_DEPENDS LoopFramer ${LLVM_TEST_DEPENDS} PARENT_SCOPE)
