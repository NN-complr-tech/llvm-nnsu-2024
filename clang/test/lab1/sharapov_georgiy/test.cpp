// RUN: %clang++ -cc1 -load %llvmshlibdir/Plugin.so -plugin myPlugin %s 2>&1 | FileCheck %s

[[deprecated]]
void oldSum(int a, int b) {}

void newSum(int a, int b) {}

class test{
 public:
    [[deprecated]]
    void oldFunc();

    void newFunc();
 private:
    [[deprecated]]
    void privOldFunc();

    void privNewFunc();
};

// CHECK: test.cpp:4:6: warning: function 'oldSum' is deprecated
// CHECK: test.cpp:11:10: warning: function 'oldFunc' is deprecated
// CHECK: test.cpp:16:10: warning: function 'privOldFunc' is deprecated
