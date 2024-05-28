; RUN: opt -load-pass-plugin=%llvmshlibdir/kalinin_loop_pass_plugin%shlibext -passes=loop_func -S %s | FileCheck %s

; int summa()
; {
;     int summa = 0;
;     for (int i = 0; i < 10; i++)
;     {
;         summa += i;
;     }
;     return summa;
; }

define dso_local i32 @summa() #0 {
entry:
  %summa = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, ptr %summa, align 4
  store i32 0, ptr %i, align 4
; CHECK: call void @loop_start()
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 10
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, ptr %i, align 4
  %2 = load i32, ptr %summa, align 4
  %add = add nsw i32 %2, %1
  store i32 %add, ptr %summa, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %3 = load i32, ptr %i, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
; CHECK: call void @loop_end()
  %4 = load i32, ptr %summa, align 4
  ret i32 %4
}
