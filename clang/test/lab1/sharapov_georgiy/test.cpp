// RUN: %clang++ -cc1 -load %llvmshlibdir/Plugin.so -plugin myPlugin %s 2>&1 | FileCheck %s

// CHECK: test.cpp:5:6: warning: function 'oldSum' is deprecated
[[deprecated]]
void oldSum(int a, int b) {}

void newSum(int a, int b) {}

class test{
 public:
    // CHECK: test.cpp:13:10: warning: function 'oldFunc' is deprecated
    [[deprecated]]
    void oldFunc();

    void newFunc();
 private:
    // CHECK: test.cpp:19:10: warning: function 'privOldFunc' is deprecated
    [[deprecated]]
    void privOldFunc();

    void privNewFunc();
};
