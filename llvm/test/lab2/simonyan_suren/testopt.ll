; ModuleID = 'D:\GitRepos\llvm-nnsu-2024\llvm\test\lab2\simonyan_suren\test.ll'
source_filename = "D:\\GitRepos\\llvm-nnsu-2024\\llvm\\test\\lab2\\simonyan_suren\\test.ll"

define dso_local void @_Z4foo1v() {
entry:
  %a = alloca float, align 4
  store float 1.000000e+00, ptr %a, align 4
  %0 = load float, ptr %a, align 4
  %add = fadd float %0, 1.000000e+00
  store float %add, ptr %a, align 4
  ret void
}

define dso_local void @_Z4bar1v() {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %0 = alloca float, align 4
  store float 1.000000e+00, ptr %0, align 4
  %1 = load float, ptr %0, align 4
  %2 = fadd float %1, 1.000000e+00
  store float %2, ptr %0, align 4
  %3 = load i32, ptr %a, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, ptr %a, align 4
  ret void
}

define dso_local void @_Z4foo2i(i32 noundef %0) {
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

define dso_local void @_Z4bar2v() {
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

define dso_local noundef float @_Z4foo3v() {
entry:
  %a = alloca float, align 4
  store float 1.000000e+00, ptr %a, align 4
  %0 = load float, ptr %a, align 4
  %add = fadd float %0, 1.000000e+00
  store float %add, ptr %a, align 4
  %1 = load float, ptr %a, align 4
  ret float %1
}

define dso_local void @_Z4bar3v() {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %call = call noundef float @_Z4foo3v()
  %0 = load i32, ptr %a, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr %a, align 4
  ret void
}

define dso_local void @_Z4foo4v() {
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

define dso_local void @_Z4bar4v() {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %0 = load i32, ptr %a, align 4
  %cmp = icmp slt i32 %0, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = alloca float, align 4
  store float 1.000000e+00, ptr %1, align 4
  %2 = load float, ptr %1, align 4
  %3 = fcmp olt float %2, 2.000000e+00
  %4 = load float, ptr %1, align 4
  %5 = fadd float %4, 1.000000e+00
  store float %5, ptr %1, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %6 = load i32, ptr %a, align 4
  %inc = add nsw i32 %6, 1
  store i32 %inc, ptr %a, align 4
  ret void
}
