# set(LLVM_LOOPFRAMER_LINK_INTO_TOOLS ON)

add_llvm_pass_plugin(LoopFramer LoopFramer.cpp)

set(LLVM_TEST_DEPENDS LoopFramer ${LLVM_TEST_DEPENDS} PARENT_SCOPE)
