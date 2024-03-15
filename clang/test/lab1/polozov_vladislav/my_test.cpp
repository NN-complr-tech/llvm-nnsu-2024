// RUN: %clang_cc1 -load %llvmshlibdir/PluginWarningDeprecated%pluginext -plugin warning-deprecated %s 2>&1 | FileCheck %s --check-prefix=DEPRECATED

// DEPRECATED: warning: find 'deprecated' in function name
void deprecated_f(int a, int b){
    int c = a + b;
}

// DEPRECATED-NOT: warning: find 'deprecated' in function name
void Matrix_Mult(){
    ;
};

// DEPRECATED-NOT: warning: find 'deprecated' in function name
void Deprecated(int c){
    int a = c - 10;
    int b = a + c;
};

// DEPRECATED: warning: find 'deprecated' in function name
int function_with_deprecated(int a, int b){
    return a - b;
}

// RUN: %clang_cc1 -load %llvmshlibdir/PluginWarningDeprecated%pluginext -plugin warning-deprecated %s -plugin-arg-warning-deprecated help 2>&1 | FileCheck %s --check-prefix=HELP

// HELP: Plugin Warning Deprecated prints a warning if a function name contains 'deprecated'