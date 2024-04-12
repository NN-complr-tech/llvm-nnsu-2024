; RUN: opt -passes=simonyan-inlining -S %s | FileCheck %s

;void foo1() {
;  float a = 1.0f;
;  a += 1.0f;
;}
;
;void bar1() {
;  int a = 0;
;  foo1();
;  a++;
;}

define dso_local void @_Z4foo1v() #0 {
entry:
  %a = alloca float, align 4
  store float 1.000000e+00, ptr %a, align 4
  %0 = load float, ptr %a, align 4
  %add = fadd float %0, 1.000000e+00
  store float %add, ptr %a, align 4
  ret void
}

define dso_local void @_Z4bar1v() #0 {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  call void @_Z4foo1v()
  %0 = load i32, ptr %a, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr %a, align 4
  ret void
}

; CHECK: define dso_local void @_Z4bar1v() {
; CHECK-NEXT:entry:
; CHECK-NEXT:  %a = alloca i32, align 4
; CHECK-NEXT:  store i32 0, ptr %a, align 4
; CHECK-NEXT:  %0 = alloca float, align 4
; CHECK-NEXT:  store float 1.000000e+00, ptr %0, align 4
; CHECK-NEXT:  %1 = load float, ptr %0, align 4
; CHECK-NEXT:  %2 = fadd float %1, 1.000000e+00
; CHECK-NEXT:  store float %2, ptr %0, align 4
; CHECK-NEXT:  %3 = load i32, ptr %a, align 4
; CHECK-NEXT:  %inc = add nsw i32 %3, 1
; CHECK-NEXT:  store i32 %inc, ptr %a, align 4
; CHECK-NEXT:  ret void
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

define dso_local void @_Z4foo2i(i32 noundef %0) #0 {
entry:
  %.addr = alloca i32, align 4
  %a = alloca float, align 4
  store i32 %0, ptr %.addr, align 4
  store float 1.000000e+00, ptr %a, align 4
  %1 = load float, ptr %a, align 4
  %add = fadd float %1, 1.000000e+00
  store float %add, ptr %a, align 4
  ret void
}

define dso_local void @_Z4bar2v() #0 {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %0 = load i32, ptr %a, align 4
  call void @_Z4foo2i(i32 noundef %0)
  %1 = load i32, ptr %a, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr %a, align 4
  ret void
}

; CHECK: define dso_local void @_Z4bar2v() {
; CHECK-NEXT: entry:
; CHECK-NEXT:  %a = alloca i32, align 4
; CHECK-NEXT:  store i32 0, ptr %a, align 4
; CHECK-NEXT:  %0 = load i32, ptr %a, align 4
; CHECK-NEXT:  call void @_Z4foo2i(i32 noundef %0)
; CHECK-NEXT:  %1 = load i32, ptr %a, align 4
; CHECK-NEXT:  %inc = add nsw i32 %1, 1
; CHECK-NEXT:  store i32 %inc, ptr %a, align 4
; CHECK-NEXT:  ret void
; CHECK-NEXT: }

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

define dso_local noundef float @_Z4foo3v() #0 {
entry:
  %a = alloca float, align 4
  store float 1.000000e+00, ptr %a, align 4
  %0 = load float, ptr %a, align 4
  %add = fadd float %0, 1.000000e+00
  store float %add, ptr %a, align 4
  %1 = load float, ptr %a, align 4
  ret float %1
}

define dso_local void @_Z4bar3v() #0 {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %call = call noundef float @_Z4foo3v()
  %0 = load i32, ptr %a, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr %a, align 4
  ret void
}

; CHECK: define dso_local void @_Z4bar3v() {
; CHECK-NEXT:entry:
; CHECK-NEXT:  %a = alloca i32, align 4
; CHECK-NEXT:  store i32 0, ptr %a, align 4
; CHECK-NEXT:  %call = call noundef float @_Z4foo3v()
; CHECK-NEXT:  %0 = load i32, ptr %a, align 4
; CHECK-NEXT:  %inc = add nsw i32 %0, 1
; CHECK-NEXT:  store i32 %inc, ptr %a, align 4
; CHECK-NEXT:  ret void
; CHECK-NEXT:}

;void foo4() {
;  float a = 1.0f;
;  if (a < 2.0f)
;    a += 1.0f;
;}
;
;void bar4() {
;  int a = 0;
;  if (a < 1)
;    foo4();
;  a++;
;}

define dso_local void @_Z4foo4v() #0 {
entry:
  %a = alloca float, align 4
  store float 1.000000e+00, ptr %a, align 4
  %0 = load float, ptr %a, align 4
  %cmp = fcmp olt float %0, 2.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = load float, ptr %a, align 4
  %add = fadd float %1, 1.000000e+00
  store float %add, ptr %a, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define dso_local void @_Z4bar4v() #0 {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %0 = load i32, ptr %a, align 4
  %cmp = icmp slt i32 %0, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @_Z4foo4v()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %1 = load i32, ptr %a, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr %a, align 4
  ret void
}

; CHECK: define dso_local void @_Z4bar4v() {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a = alloca i32, align 4
; CHECK-NEXT:   store i32 0, ptr %a, align 4
; CHECK-NEXT:   %0 = load i32, ptr %a, align 4
; CHECK-NEXT:   %cmp = icmp slt i32 %0, 1
; CHECK-NEXT:   br i1 %cmp, label %if.then, label %if.end
; CHECK: if.then:                                          ; preds = %entry
; CHECK-NEXT:   %1 = alloca float, align 4
; CHECK-NEXT:   store float 1.000000e+00, ptr %1, align 4
; CHECK-NEXT:   %2 = load float, ptr %1, align 4
; CHECK-NEXT:   %3 = fcmp olt float %2, 2.000000e+00
; CHECK-NEXT:   %4 = load float, ptr %1, align 4
; CHECK-NEXT:   %5 = fadd float %4, 1.000000e+00
; CHECK-NEXT:   store float %5, ptr %1, align 4
; CHECK-NEXT:   br label %if.end
; CHECK: if.end:                                           ; preds = %if.then, %entry
; CHECK-NEXT:   %6 = load i32, ptr %a, align 4
; CHECK-NEXT:   %inc = add nsw i32 %6, 1
; CHECK-NEXT:   store i32 %inc, ptr %a, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }