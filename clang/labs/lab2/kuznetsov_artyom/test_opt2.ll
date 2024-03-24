; ModuleID = 'test2.ll'
source_filename = "test2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@j = dso_local global i32 2, align 4

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @a() #0 {
entry:
  call void @instrument_start()
  %c = alloca i32, align 4
  %i = alloca i32, align 4
  %q = alloca i32, align 4
  store i32 10, ptr %c, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 10
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, ptr %c, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr %c, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %2 = load i32, ptr %i, align 4
  %inc1 = add nsw i32 %2, 1
  store i32 %inc1, ptr %i, align 4
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %for.cond
  %3 = load i32, ptr %c, align 4
  %add = add nsw i32 %3, 42
  store i32 %add, ptr %q, align 4
  call void @instrument_end()
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @b(i32 noundef %a) #0 {
entry:
  call void @instrument_start()
  %a.addr = alloca i32, align 4
  %c = alloca i32, align 4
  %q = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 10, ptr %c, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr @j, align 4
  %1 = load i32, ptr %a.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32, ptr %c, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, ptr %c, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %3 = load i32, ptr @j, align 4
  %inc1 = add nsw i32 %3, 1
  store i32 %inc1, ptr @j, align 4
  br label %for.cond, !llvm.loop !8

for.end:                                          ; preds = %for.cond
  %4 = load i32, ptr %c, align 4
  %add = add nsw i32 %4, 42
  store i32 %add, ptr %q, align 4
  %5 = load i32, ptr %q, align 4
  call void @instrument_end()
  ret i32 %5
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @v() #0 {
entry:
  call void @instrument_start()
  %c = alloca i32, align 4
  %q = alloca i32, align 4
  store i32 10, ptr %c, align 4
  call void @a()
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  br label %for.end

for.inc:                                          ; No predecessors!
  %0 = load i32, ptr @j, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @j, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %1 = load i32, ptr %c, align 4
  %call = call i32 @b(i32 noundef %1)
  %2 = load i32, ptr %c, align 4
  %add = add nsw i32 %2, 42
  store i32 %add, ptr %q, align 4
  %3 = load i32, ptr %q, align 4
  call void @instrument_end()
  ret i32 %3
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
entry:
  call void @instrument_start()
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  call void @instrument_end()
  ret i32 0
}

declare void @instrument_start()

declare void @instrument_end()

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 17.0.6 (https://github.com/Kuznetsov-Artyom/llvm-nnsu-2024 03f801606a14fbfaadc8e1465b2bc7e1f8dd2e24)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
