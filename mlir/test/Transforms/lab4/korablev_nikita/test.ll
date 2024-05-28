; ModuleID = '/home/nikita/Repositories/UNN_labs/llvm-project/mlir/test/Transforms/lab4/korablev_nikita/test.cpp'
source_filename = "/home/nikita/Repositories/UNN_labs/llvm-project/mlir/test/Transforms/lab4/korablev_nikita/test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z3sumii(i32 noundef %a, i32 noundef %b) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z4multii(i32 noundef %a, i32 noundef %b) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %mul = mul nsw i32 %0, %1
  ret i32 %mul
}

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #1 {
entry:
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 5, ptr %a, align 4
  %0 = load i32, ptr %a, align 4
  %call = call noundef i32 @_Z3sumii(i32 noundef %0, i32 noundef 2)
  store i32 %call, ptr %a, align 4
  %1 = load i32, ptr %a, align 4
  %call1 = call noundef i32 @_Z4multii(i32 noundef %1, i32 noundef 5)
  store i32 %call1, ptr %a, align 4
  %2 = load i32, ptr %a, align 4
  %3 = load i32, ptr %a, align 4
  %call2 = call noundef i32 @_Z3sumii(i32 noundef %2, i32 noundef %3)
  store i32 %call2, ptr %a, align 4
  ret i32 0
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline norecurse nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 17.0.6 (git@github.com:NikitaKorablev/llvm-project.git 28fa8528faba5cf51c4696f73070163d53f80f00)"}
