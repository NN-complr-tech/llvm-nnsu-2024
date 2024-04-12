; ModuleID = 'd:\GitRepos\llvm-nnsu-2024\llvm\test\lab2\simonyan_suren\testfun.cpp'
source_filename = "d:\\GitRepos\\llvm-nnsu-2024\\llvm\\test\\lab2\\simonyan_suren\\testfun.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
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

; Function Attrs: mustprogress noinline nounwind optnone uwtable
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

attributes #0 = { mustprogress noinline nounwind optnone uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{i32 1, !"MaxTLSAlign", i32 65536}
!4 = !{!"clang version 17.0.6 (https://github.com/SSuren4ik/llvm-nnsu-2024.git 7d2f0dadb28ce8c42178a65b334c1d968eeb9ffa)"}
