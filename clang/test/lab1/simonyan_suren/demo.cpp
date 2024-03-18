// RUN: %clang_cc1 -load %llvmshlibdir/AlwaysInlinePlugin%pluginext -plugin always-inlines-plugin %s 2>&1 | FileCheck %s

// CHECK: Added attribute always_inline in sum
int sum(int A, int B) { return A + B; }

// CHECK: Don't added attribute always_inline in checkIf
void checkIf(int A, int B) {
  {
    if (A > B) {
      A = B;
    }
  }
}

// CHECK: Don't added attribute always_inline in checkSwitch
void checkSwitch(int A, int B) {
  switch(A) {
    case 0:
    break;
    default: 
    break;
  }
}

// CHECK: Added attribute always_inline in checkEmpty
void checkEmpty() {}

// CHECK: Don't added attribute always_inline in checkWhile
void checkWhile() {
  while (true) {
  }
}

// CHECK: Added attribute always_inline in checkSkobki
void checkSkobki() {
  {
    {
      {
        {;}
      }
    }
  }
}

// CHECK: Don't added attribute always_inline in checkFor
void checkFor(int n) {
    for (int i = 0; i < n; i++) {
        int x = i;
    }
}

// CHECK: Don't added attribute always_inline in checkDoWhile
void checkDoWhile(int n) {
    int i = 0;
    do {
        i++;
    } while (i < n);
}
