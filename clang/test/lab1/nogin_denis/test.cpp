// RUN: %clang_cc1 -load %llvmshlibdir/AddAlwaysInlinePlugin%pluginext -plugin add-always-inline %s 1>&1 | FileCheck %s

// CHECK: Skip (always_inline already exists): empty
__attribute__((always_inline)) void empty() {
    ;
}

// CHECK: Add always_inline to function: simple
int simple(int a) {
    return a;
}

// CHECK: Add always_inline to function: sum
int sum(int a, int b) {
    {
        return a + b;
    }
}

// CHECK: Add always_inline to function: assign
void assign(int a, int b) {
    {}
    {
        {
            a = b;
        }
    }
}

// CHECK: Skip (condition found): whileTest
void whileTest(int a, int b) {
    while (a < b) {
        a += b;
    }
}

// CHECK: Skip (condition found): ifTest
void ifTest(int a, int b) {
    {
        if (a > b) {
            a -= b;
        }
    }
}

// CHECK: Skip (condition found): forTest
void forTest(int a) {
    {}
    for (int i = 0; i < a; ++i) {
        a += i;
    }
}

// CHECK: Skip (condition found): switchTest
int switchTest(int a, int b) {
    {{}{{}}}
    switch(a) {
        case 1:
            return a;
        case 2:
            return b;
    }
    {}
    return 0;
}