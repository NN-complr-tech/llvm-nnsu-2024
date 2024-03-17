// RUN: d:\gitrepos\llvm-nnsu-2024\build\bin\clang.exe -cc1 -internal-isystem d:/gitrepos/llvm-nnsu-2024/build/lib/clang/17/include -nostdsysteminc -load D:\GitRepos\llvm-nnsu-2024\build\bin/AlwaysInlinePlugin.dll -plugin always-inlines-plugin %s | d:\gitrepos\llvm-nnsu-2024\build\bin\filecheck.exe %s

// CHECK: Added attribute always_inline in sum
int sum(int A, int B) { return A + B; }

// CHECK: checkAlw not suitable for the attribute
void checkAlw(int A, int B) {
  {
    if (A > B) {
      A = B;
    }
  }
}

// CHECK: Added attribute always_inline in checkEmpty
void checkEmpty() {}

// CHECK: ifEqual not suitable for the attribute
void ifEqual(char A, char B) {
  {
    {
      {
        {
          while (true) {
          }
          {}
        }
      }
    }
  }
}