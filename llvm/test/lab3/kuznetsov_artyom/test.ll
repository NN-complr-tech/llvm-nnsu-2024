; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none) uwtable
define dso_local noundef <2 x double> @_Z12muladd_test1Dv2_dS_S_(<2 x double> noundef %a, <2 x double> noundef %b, <2 x double> noundef %c) local_unnamed_addr #0 {
entry:
  %0 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c)
  ret <2 x double> %0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef <2 x double> @_Z12muladd_test2Dv2_dS_S_(<2 x double> noundef %a, <2 x double> noundef %b, <2 x double> noundef %c) local_unnamed_addr #2 {
entry:
  %mul = fmul <2 x double> %a, %b
  %add = fadd <2 x double> %mul, %c
  ret <2 x double> %add
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none) uwtable
define dso_local noundef <2 x double> @_Z12muladd_test3Dv2_dS_S_(<2 x double> noundef %a, <2 x double> noundef %b, <2 x double> noundef %c) local_unnamed_addr #0 {
entry:
  %0 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %a, <2 x double> %c, <2 x double> %b)
  %1 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %0, <2 x double> %c, <2 x double> %b)
  ret <2 x double> %1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef <2 x double> @_Z12muladd_test4Dv2_dS_S_(<2 x double> noundef %a, <2 x double> noundef %b, <2 x double> noundef %c) local_unnamed_addr #2 {
entry:
  %mul = fmul <2 x double> %a, %c
  %div = fdiv <2 x double> %mul, %b
  %add = fadd <2 x double> %div, %b
  ret <2 x double> %add
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef <2 x double> @_Z12muladd_test5Dv2_dS_S_(<2 x double> noundef %a, <2 x double> noundef %b, <2 x double> noundef %c) local_unnamed_addr #2 {
entry:
  %mul = fmul <2 x double> %a, %b
  %add = fadd <2 x double> %a, %b
  %sub = fsub <2 x double> %add, %c
  %add1 = fadd <2 x double> %mul, %c
  %add2 = fadd <2 x double> %add1, %sub
  ret <2 x double> %add2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef <2 x double> @_Z12muladd_test6Dv2_dS_S_(<2 x double> noundef %a, <2 x double> noundef %b, <2 x double> noundef %c) local_unnamed_addr #2 {
entry:
  %mul = fmul <2 x double> %a, %b
  %mul1 = fmul <2 x double> %mul, %c
  ret <2 x double> %mul1
}

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="128" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="128" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 17.0.6 (https://github.com/Kuznetsov-Artyom/llvm-nnsu-2024 d0378b668d4e9e7b60a19069049f55c257d8a3bf)"}
