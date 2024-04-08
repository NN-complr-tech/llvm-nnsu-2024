; RUN: opt -load-pass-plugin=%llvmshlibdir/SimonyanInliningPass%shlibext -passes=simonyan-inlining -S %s | FileCheck %s

;void foo1() {
;  float a = 1.0f;
;  a += 1.0f;
;}
;
;void bar1() {
;  int a = 0;
;  foo();
;  a++;
;}

define dso_local void @_Z3foo1v() {
  %1 = alloca float, align 4
  store float 1.000000e+00, float* %1, align 4
  %2 = load float, float* %1, align 4
  %3 = fadd float %2, 1.000000e+00
  store float %3, float* %1, align 4
  ret void
}

define dso_local void @_Z3bar1v() {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @_Z3foo1v()
  %2 = load i32, i32* %1, align 4
  %3 = add nsw i32 %2, 1
  store i32 %3, i32* %1, align 4
  ret void
}

; CHECK: define dso_local void @_Z3bar1v() {
; CHECK-NEXT: %1 = alloca i32, align 4
; CHECK-NEXT: store i32 0, ptr %1, align 4
; CHECK-NEXT: %2 = alloca float, align 4
; CHECK-NEXT: store float 1.000000e+00, ptr %2, align 4
; CHECK-NEXT: %3 = load float, ptr %2, align 4
; CHECK-NEXT: %4 = fadd float %3, 1.000000e+00
; CHECK-NEXT: store float %4, ptr %2, align 4
; CHECK-NEXT: %5 = load i32, ptr %1, align 4
; CHECK-NEXT: %6 = add nsw i32 %5, 1
; CHECK-NEXT: store i32 %6, ptr %1, align 4
; CHECK-NEXT: ret void
; CHECK-NEXT: }

;void foo2(int) {
;  float a = 1.0f;
;  a += 1.0f;
;}
;
;void bar2() {
;  int a = 0;
;  foo2(a);
;  a++;
;}

define dso_local void @_Z3foo2i(i32 noundef %0) {
  %2 = alloca i32, align 4
  %3 = alloca float, align 4
  store i32 %0, i32* %2, align 4
  store float 1.000000e+00, float* %3, align 4
  %4 = load float, float* %3, align 4
  %5 = fadd float %4, 1.000000e+00
  store float %5, float* %3, align 4
  ret void
}

define dso_local void @_Z3bar2v() {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = load i32, i32* %1, align 4
  call void @_Z3foo2i(i32 noundef %2)
  %3 = load i32, i32* %1, align 4
  %4 = add nsw i32 %3, 1
  store i32 %4, i32* %1, align 4
  ret void
}


; CHECK: define dso_local void @_Z3bar2v() {
; CHECK-NEXT: %1 = alloca i32, align 4
; CHECK-NEXT: store i32 0, ptr %1, align 4
; CHECK-NEXT: %2 = load i32, ptr %1, align 4
; CHECK-NEXT: call void @_Z3foo2i(i32 noundef %2)
; CHECK-NEXT: %3 = load i32, ptr %1, align 4
; CHECK-NEXT: %4 = add nsw i32 %3, 1
; CHECK-NEXT: store i32 %4, ptr %1, align 4
; CHECK-NEXT: ret void
; CHECK-NEXT:}

;float foo3() {
;  float a = 1.0f;
;  a += 1.0f;
;  return a;
;}
;
;void bar3() {
;  int a = 0;
;  foo3();
;  a++;
;}

define dso_local noundef float @_Z3foo3v() {
  %1 = alloca float, align 4
  store float 1.000000e+00, float* %1, align 4
  %2 = load float, float* %1, align 4
  %3 = fadd float %2, 1.000000e+00
  store float %3, float* %1, align 4
  %4 = load float, float* %1, align 4
  ret float %4
}

define dso_local void @_Z3bar3v() {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = call noundef float @_Z3foo3v()
  %3 = load i32, i32* %1, align 4
  %4 = add nsw i32 %3, 1
  store i32 %4, i32* %1, align 4
  ret void
}

; CHECK: define dso_local void @_Z3bar3v() {
; CHECK-NEXT: %1 = alloca i32, align 4
; CHECK-NEXT: store i32 0, ptr %1, align 4
; CHECK-NEXT: %2 = call noundef float @_Z3foo3v()
; CHECK-NEXT: %3 = load i32, ptr %1, align 4
; CHECK-NEXT: %4 = add nsw i32 %3, 1
; CHECK-NEXT: store i32 %4, ptr %1, align 4
; CHECK-NEXT: ret void
; CHECK-NEXT: }

;void foo4() {
;  float a = 1.0f;
;  a += 1.0f;
;  foo();
;}
;
;void bar4() {
;  int a = 0;
;  foo();
;  a++;
;}


define dso_local void @_Z3foo4v() {
  %1 = alloca float, align 4
  store float 1.000000e+00, float* %1, align 4
  %2 = load float, float* %1, align 4
  %3 = fadd float %2, 1.000000e+00
  store float %3, float* %1, align 4
  call void @_Z3foo4v()
  ret void
}

define dso_local void @_Z3bar4v() { 
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = call noundef float @_Z3foo4v()
  %3 = load i32, i32* %1, align 4
  %4 = add nsw i32 %3, 1
  store i32 %4, i32* %1, align 4
  ret void
}

; CHECK: define dso_local void @_Z3bar4v() {
; CHECK-NEXT: %1 = alloca i32, align 4
; CHECK-NEXT: store i32 0, ptr %1, align 4
; CHECK-NEXT: %2 = alloca float, align 4
; CHECK-NEXT: store float 1.000000e+00, ptr %2, align 4
; CHECK-NEXT: %3 = load float, ptr %2, align 4
; CHECK-NEXT: %4 = fadd float %3, 1.000000e+00
; CHECK-NEXT: store float %4, ptr %2, align 4
; CHECK-NEXT: call void @_Z3foo4v()
; CHECK-NEXT: %5 = load i32, ptr %1, align 4
; CHECK-NEXT: %6 = add nsw i32 %5, 1
; CHECK-NEXT: store i32 %6, ptr %1, align 4
; CHECK-NEXT: ret void
; CHECK-NEXT: }