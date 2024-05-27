// RUN: %clang_cc1 -load %llvmshlibdir/deprWarnPluginKalinin%pluginext -plugin plugin_for_deprecated_functions %s 2>&1 | FileCheck %s

// CHECK: warning: The function name has 'deprecated'
void deprecated2();

// CHECK: warning: The function name has 'deprecated'
void deprecatedFunc2();

// CHECK: warning: The function name has 'deprecated'
int deprecatedSumm2(int a, int b) {
    return a + b;
}

// CHECK-NOT: warning: The function name has 'deprecated'
void deprecation2();

// CHECK-NOT: warning: The function name has 'deprecated'
void deprfunction2();

// CHECK-NOT: warning: The function name has 'deprecated'
void foo2();

// CHECK: warning: The function name has 'deprecated'
void deprecated();

// CHECK: warning: The function name has 'deprecated'
void deprecatedFunc();

// CHECK: warning: The function name has 'deprecated'
int deprecatedSumm(int a, int b) {
    return a + b;
}

// CHECK-NOT: warning: The function name has 'deprecated'
void deprecation();

// CHECK-NOT: warning: The function name has 'deprecated'
void deprfunction();

// CHECK-NOT: warning: The function name has 'deprecated'
void foo();

class Test {
    // CHECK: warning: The function name has 'deprecated'
    void is_deprecated_function();
    // CHECK-NOT: warning: The function name has 'deprecated'
    void depfunc();
};