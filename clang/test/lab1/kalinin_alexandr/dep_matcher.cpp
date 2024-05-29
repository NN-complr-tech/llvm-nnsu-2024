// RUN: %clang_cc1 -load %llvmshlibdir/deprWarnPluginKalinin%pluginext -plugin plugin_for_deprecated_functions -case-sensitive %s 2>&1 | FileCheck %s --check-prefix=CASE-SENSITIVE
// RUN: %clang_cc1 -load %llvmshlibdir/deprWarnPluginKalinin%pluginext -plugin plugin_for_deprecated_functions -case-insensitive %s 2>&1 | FileCheck %s --check-prefix=CASE-INSENSITIVE


// CASE-SENSITIVE: warning: The function name has 'deprecated'
void deprecated();

// CASE-SENSITIVE: warning: The function name has 'deprecated'
void deprecatedFunc();

// CASE-SENSITIVE: warning: The function name has 'deprecated'
int deprecatedSumm(int a, int b) {
    return a + b;
}

// CASE-SENSITIVE-NOT: warning: The function name has 'deprecated'
void deprecation();

// CASE-SENSITIVE-NOT: warning: The function name has 'deprecated'
void deprfunction();

// CASE-SENSITIVE-NOT: warning: The function name has 'deprecated'
void foo();

class Test {
    // CASE-SENSITIVE: warning: The function name has 'deprecated'
    void is_deprecated_function();
    // CASE-SENSITIVE-NOT: warning: The function name has 'deprecated'
    void depfunc();
};

// CASE-SENSITIVE-NOT: warning: The function name has 'deprecated'
void DEPRECATED_FUNC();

// CASE-SENSITIVE-NOT: warning: The function name has 'deprecated'
void DePrEcAtEd_FuNc();

// CASE-SENSITIVE-NOT: warning: The function name has 'deprecated'
int DEPRECATED_SUMM(int a, int b) {
    return a + b;
}
// CASE-SENSITIVE-NOT: warning: The function name has 'deprecated'
void dEpReCaTiOn();

// CASE-SENSITIVE-NOT: warning: The function name has 'deprecated'
void DePrFuNc();


// CASE-INSENSITIVE: warning: The function name has 'deprecated'
void deprecated();

// CASE-INSENSITIVE: warning: The function name has 'deprecated'
void deprecatedFunc();

// CASE-INSENSITIVE: warning: The function name has 'deprecated'
int deprecatedSumm(int a, int b) {
    return a + b;
}
// CASE-INSENSITIVE: warning: The function name has 'deprecated'
void deprecation();

// CASE-INSENSITIVE: warning: The function name has 'deprecated'
void deprfunction();

// CASE-INSENSITIVE-NOT: warning: The function name has 'deprecated'
void foo();

class Test {
    // CASE-INSENSITIVE: warning: The function name has 'deprecated'
    void is_deprecated_function();
    // CASE-INSENSITIVE-NOT: warning: The function name has 'deprecated'
    void depfunc();
};

// CASE-INSENSITIVE: warning: The function name has 'deprecated'
void DEPRECATED_FUNC();

// CASE-INSENSITIVE: warning: The function name has 'deprecated'
void DePrEcAtEd_FuNc();

// CASE-INSENSITIVE: warning: The function name has 'deprecated'
int DEPRECATED_SUMM(int a, int b) {
    return a + b;
}
// CASE-INSENSITIVE: warning: The function name has 'deprecated'
void dEpReCaTiOn();

// CASE-INSENSITIVE: warning: The function name has 'deprecated'
void DePrFuNc();