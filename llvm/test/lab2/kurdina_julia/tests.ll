; RUN: opt -load-pass-plugin=%llvmshlibdir/kurdina_loop_pass_plugin%shlibext -passes=loop_func -S %s | FileCheck %s

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

; int minus(int a, int b) {
; 	while (a < b) {
; 		a -= b;
; 	}
; 	return a;
; } 

define dso_local i32 @minus(i32 noundef %a, i32 noundef %b) #0 {
entry:
  a.addr = alloca i32, align 4
  b.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
; CHECK: call void @loop_start()
  br label %while.cond

while.cond:                                        ; preds = %while.body, %entry
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %while.body, label %while.end

while.body:                                        ; preds = %while.cond
  %2 = load i32, ptr %a.addr, align 4
  %3 = load i32, ptr %a.addr, align 4
  %sub = sub nsw i32 %2, %3
  store i32 %sub, ptr %a_ptr, align 4
  br label %while.cond

while.end:                                         ; preds = %while.cond
; CHECK: call void @loop_end()
  %4 = load i32, ptr %a_ptr, align 4
  ret i32 %4
}