if (NOT WIN32 AND NOT CYGWIN)
    add_llvm_pass_plugin(PolozovLoopPass LoopPass.cpp)
    set(LLVM_TEST_DEPENDS PolozovLoopPass ${LLVM_TEST_DEPENDS} PARENT_SCOPE)
endif()
