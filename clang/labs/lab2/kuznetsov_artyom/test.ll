; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local noundef i32 @_Z4facti(i32 noundef %value) #0 {
entry:
  %value.addr = alloca i32, align 4
  store i32 %value, ptr %value.addr, align 4
  %0 = load i32, ptr %value.addr, align 4
  %cmp = icmp sle i32 %0, 1
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  br label %cond.end

cond.false:                                       ; preds = %entry
  %1 = load i32, ptr %value.addr, align 4
  %2 = load i32, ptr %value.addr, align 4
  %sub = sub nsw i32 %2, 1
  %call = call noundef i32 @_Z4facti(i32 noundef %sub)
  %mul = mul nsw i32 %1, %call
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ 1, %cond.true ], [ %mul, %cond.false ]
  ret i32 %cond
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z4funcb(i1 noundef zeroext %pred) #1 {
entry:
  %retval = alloca i32, align 4
  %pred.addr = alloca i8, align 1
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %a1 = alloca i32, align 4
  %b2 = alloca i32, align 4
  %c = alloca i32, align 4
  %frombool = zext i1 %pred to i8
  store i8 %frombool, ptr %pred.addr, align 1
  %0 = load i8, ptr %pred.addr, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 10, ptr %a, align 4
  store i32 20, ptr %b, align 4
  %1 = load i32, ptr %a, align 4
  %2 = load i32, ptr %b, align 4
  %add = add nsw i32 %1, %2
  store i32 %add, ptr %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  store i32 100, ptr %a1, align 4
  store i32 200, ptr %b2, align 4
  store i32 300, ptr %c, align 4
  %3 = load i32, ptr %a1, align 4
  %4 = load i32, ptr %b2, align 4
  %add3 = add nsw i32 %3, %4
  %5 = load i32, ptr %c, align 4
  %add4 = add nsw i32 %add3, %5
  store i32 %add4, ptr %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %6 = load i32, ptr %retval, align 4
  ret i32 %6
}

attributes #0 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 17.0.6 (https://github.com/Kuznetsov-Artyom/llvm-nnsu-2024 d7be4ae16f8e8fdd1468dd1e2ea9e030c755f64d)"}
