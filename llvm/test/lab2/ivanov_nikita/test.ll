; RUN: opt -passes=ivanov-loop-pass -S %s 2>&1 | FileCheck %s

; void foo(int n, int m) {
;     int c0;
;     int c1;
;     for (c0 = n; c0 > 0; c0--) {
;         c1++;
;     }
; }

define dso_local void @_Z3fooii(i32 noundef %n, i32 noundef %m) {
entry:
  %n.addr = alloca i32, align 4
  %m.addr = alloca i32, align 4
  %c0 = alloca i32, align 4
  %c1 = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  store i32 %m, ptr %m.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  store i32 %0, ptr %c0, align 4
; CHECK:    call void @loop_start()
  br label %for.cond

for.cond:
  %1 = load i32, ptr %c0, align 4
  %cmp = icmp sgt i32 %1, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %2 = load i32, ptr %c1, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, ptr %c1, align 4
  br label %for.inc

for.inc:
  %3 = load i32, ptr %c0, align 4
  %dec = add nsw i32 %3, -1
  store i32 %dec, ptr %c0, align 4
  br label %for.cond

for.end:
; CHECK:    call void @loop_end()
  ret void
}
