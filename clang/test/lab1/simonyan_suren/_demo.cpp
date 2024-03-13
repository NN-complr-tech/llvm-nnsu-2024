// RUN: split-file %s %t
// RUN: %clang_cc1 -load %llvmshlibdir/"AlwaysInlinePlugin"%pluginext -add-plugin always-inlines-plugin
// RUN: %clang_cc1 -load %llvmshlibdir/"AlwaysInlinePlugin"%pluginext -add-plugin always-inlines-plugin

int sum(int a, int b) { return a + b; }

void checkAlw(int a, int b) {
  if (a < b) {
    a = b;
  }
}

void fun_with_while() {
    int i = 0;
    while(i < 5) {
        i++;
    }
}

void ifEqual(char a, char b) {
  if (a != b) {
  }
}

inline int sum_with_inline(int a, int b) { return a + b; }
