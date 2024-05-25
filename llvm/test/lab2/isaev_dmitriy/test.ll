; RUN: opt -load-pass-plugin=%llvmshlibdir/IsaevInlinePass%pluginext -passes=isaev-inline -S %s | FileCheck %s

; Test1
define dso_local void @_Z4foo1v() {
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
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a = alloca i32, align 4
; CHECK-NEXT:   store i32 0, ptr %a, align 4
; CHECK-NEXT:   %1 = alloca float, align 4
; CHECK-NEXT:   store float 1.000000e+00, ptr %1, align 4
; CHECK-NEXT:   %2 = load float, ptr %1, align 4
; CHECK-NEXT:   %3 = fadd float %2, 1.000000e+00
; CHECK-NEXT:   store float %3, ptr %1, align 4
; CHECK-NEXT:   %0 = load i32, ptr %a, align 4
; CHECK-NEXT:   %inc = add nsw i32 %0, 1
; CHECK-NEXT:   store i32 %inc, ptr %a, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; Test2
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
; CHECK-NEXT:   %a = alloca i32, align 4
; CHECK-NEXT:   store i32 0, ptr %a, align 4
; CHECK-NEXT:   %0 = load i32, ptr %a, align 4
; CHECK-NEXT:   %1 = alloca float, align 4
; CHECK-NEXT:   store float 1.000000e+00, ptr %1, align 4
; CHECK-NEXT:   %2 = load float, ptr %1, align 4
; CHECK-NEXT:   %3 = fadd float %2, 1.000000e+00
; CHECK-NEXT:   store float %3, ptr %1, align 4
; CHECK-NEXT:   %4 = load i32, ptr %a, align 4
; CHECK-NEXT:   %inc = add nsw i32 %4, 1
; CHECK-NEXT:   store i32 %inc, ptr %a, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; Test3
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
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a = alloca i32, align 4
; CHECK-NEXT:   store i32 0, ptr %a, align 4
; CHECK-NEXT:   %1 = alloca float, align 4
; CHECK-NEXT:   store float 1.000000e+00, ptr %1, align 4
; CHECK-NEXT:   %2 = load float, ptr %1, align 4
; CHECK-NEXT:   %3 = fadd float %2, 1.000000e+00
; CHECK-NEXT:   store float %3, ptr %1, align 4
; CHECK-NEXT:   %call = load float, ptr %1, align 4
; CHECK-NEXT:   %0 = load i32, ptr %a, align 4
; CHECK-NEXT:   %inc = add nsw i32 %0, 1
; CHECK-NEXT:   store i32 %inc, ptr %a, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; Test4
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
; CHECK-NEXT:   br i1 %3, label %if.then.then, label %if.then.end
; CHECK: if.then.then:                                     ; preds = %if.then
; CHECK-NEXT:   %4 = load float, ptr %1, align 4
; CHECK-NEXT:   %add = fadd float %4, 1.000000e+00
; CHECK-NEXT:   store float %add, ptr %1, align 4
; CHECK-NEXT:   br label %if.then.end
; CHECK: if.then.end:                                      ; preds = %if.then.then, %if.then
; CHECK-NEXT:   br label %if.end
; CHECK: if.end:                                           ; preds = %if.then.end, %entry
; CHECK-NEXT:   %5 = load i32, ptr %a, align 4
; CHECK-NEXT:   %inc = add nsw i32 %5, 1
; CHECK-NEXT:   store i32 %inc, ptr %a, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }