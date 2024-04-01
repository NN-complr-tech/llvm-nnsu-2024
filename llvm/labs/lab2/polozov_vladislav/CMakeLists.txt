if (NOT WIN32 AND NOT CYGWIN)
    add_llvm_pass_plugin(LoopPass LoopPass.cpp)
    set(LLVM_TEST_DEPENDS LoopPass ${LLVM_TEST_DEPENDS} PARENT_SCOPE)
endif()