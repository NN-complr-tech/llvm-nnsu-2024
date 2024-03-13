// RUN: split-file %s %t
// RUN: %clang++ -cc1 -load %llvmshlibdir/DepWarningPlugin%pluginext -plugin DepEmitWarning %t/no_args.cpp 2>&1 | FileCheck %t/no_args.cpp
// RUN: %clang++ -cc1 -load %llvmshlibdir/DepWarningPlugin%pluginext -plugin DepEmitWarning -plugin-arg-DepEmitWarning --attr-style=gcc %t/gcc.cpp 2>&1 | FileCheck %t/gcc.cpp
// RUN: %clang++ -cc1 -load %llvmshlibdir/DepWarningPlugin%pluginext -plugin DepEmitWarning -plugin-arg-DepEmitWarning --attr-style=c++14 %t/c++14.cpp 2>&1 | FileCheck %t/c++14.cpp
// RUN: %clang++ -cc1 -load %llvmshlibdir/DepWarningPlugin%pluginext -plugin DepEmitWarning -plugin-arg-DepEmitWarning --attr-style=both %t/both.cpp 2>&1 | FileCheck %t/both.cpp
// RUN: %clang++ -cc1 -load %llvmshlibdir/DepWarningPlugin%pluginext -plugin DepEmitWarning -plugin-arg-DepEmitWarning --attr-style=aaaaa %t/wrong_style.cpp 2>&1 | FileCheck %t/wrong_style.cpp


//--- no_args.cpp

// CHECK: no_args.cpp:3:20: warning: function 'oldSum' is deprecated (c++14)
[[deprecated]] int oldSum(int a, int b);

// CHECK: no_args.cpp:6:6: warning: function 'oldFunc1' is deprecated (gcc)
void oldFunc1(int a, int b) __attribute__((deprecated));

// CHECK: no_args.cpp:9:34: warning: function 'oldFunc2' is deprecated (gcc)
__attribute__((deprecated)) void oldFunc2(int a, int b);

// CHECK: no_args.cpp:13:6: warning: function 'oldFunc3' is deprecated (gcc)
__attribute__((deprecated))
void oldFunc3(int a, int b);

// CHECK: no_args.cpp:17:6: warning: function 'oldFunc4' is deprecated (c++14)
[[deprecated]]
void oldFunc4(int a, int b);

// CHECK-NOT: warning: 
void Func(int a, int b);

//--- gcc.cpp

// CHECK-NOT: warning:
[[deprecated]] int oldSum(int a, int b);

// CHECK: gcc.cpp:6:6: warning: function 'oldFunc1' is deprecated (gcc)
void oldFunc1(int a, int b) __attribute__((deprecated));

// CHECK: gcc.cpp:9:34: warning: function 'oldFunc2' is deprecated (gcc)
__attribute__((deprecated)) void oldFunc2(int a, int b);

// CHECK: gcc.cpp:13:6: warning: function 'oldFunc3' is deprecated (gcc)
__attribute__((deprecated))
void oldFunc3(int a, int b);

// CHECK-NOT: warning:
[[deprecated]]
void oldFunc4(int a, int b);

// CHECK-NOT: warning: 
void Func(int a, int b);

//--- c++14.cpp

// CHECK: c++14.cpp:3:20: warning: function 'oldSum' is deprecated (c++14)
[[deprecated]] int oldSum(int a, int b);

// CHECK-NOT: warning:
void oldFunc1(int a, int b) __attribute__((deprecated));

// CHECK-NOT: warning:
__attribute__((deprecated)) void oldFunc2(int a, int b);

// CHECK-NOT: warning:
__attribute__((deprecated))
void oldFunc3(int a, int b);

// CHECK: c++14.cpp:17:6: warning: function 'oldFunc4' is deprecated (c++14)
[[deprecated]]
void oldFunc4(int a, int b);

// CHECK-NOT: warning: 
void Func(int a, int b);

//--- both.cpp

// CHECK: both.cpp:3:20: warning: function 'oldSum' is deprecated (c++14)
[[deprecated]] int oldSum(int a, int b);

// CHECK: both.cpp:6:6: warning: function 'oldFunc1' is deprecated (gcc)
void oldFunc1(int a, int b) __attribute__((deprecated));

// CHECK: both.cpp:9:34: warning: function 'oldFunc2' is deprecated (gcc)
__attribute__((deprecated)) void oldFunc2(int a, int b);

// CHECK: both.cpp:13:6: warning: function 'oldFunc3' is deprecated (gcc)
__attribute__((deprecated))
void oldFunc3(int a, int b);

// CHECK: both.cpp:17:6: warning: function 'oldFunc4' is deprecated (c++14)
[[deprecated]]
void oldFunc4(int a, int b);

// CHECK-NOT: warning:
void Func(int a, int b);

//--- wrong_style.cpp

// CHECK: wrong_style.cpp:3:20: warning: function 'oldSum' is deprecated (c++14)
[[deprecated]] int oldSum(int a, int b);

// CHECK: wrong_style.cpp:6:6: warning: function 'oldFunc1' is deprecated (gcc)
void oldFunc1(int a, int b) __attribute__((deprecated));

// CHECK: wrong_style.cpp:9:34: warning: function 'oldFunc2' is deprecated (gcc)
__attribute__((deprecated)) void oldFunc2(int a, int b);

// CHECK: wrong_style.cpp:13:6: warning: function 'oldFunc3' is deprecated (gcc)
__attribute__((deprecated))
void oldFunc3(int a, int b);

// CHECK: wrong_style.cpp:17:6: warning: function 'oldFunc4' is deprecated (c++14)
[[deprecated]]
void oldFunc4(int a, int b);

// CHECK-NOT: warning:
void Func(int a, int b);