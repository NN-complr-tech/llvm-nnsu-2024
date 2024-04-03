; RUN: opt -load-pass-plugin=%llvmshlibdir/AlexseevLoopPlugin%shlibext -passes=alexseev-loop-plugin -S %s | FileCheck %s

; void whileFoo() {
;     int i = 10;
;     while (i > 0) {
;         i--;
;         if (i == 3)
;             break;
;     }
; }

define dso_local void @whileFoo() {
entry:
  %i = alloca i32, align 4
  store i32 10, i32* %i, align 4
  
  ; CHECK: call void @loop_start()
  ; CHECK-NEXT: br label %while.cond
  
  br label %while.cond

while.cond:
  %0 = load i32, i32* %i, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %while.body, label %while.end

while.body:
  %1 = load i32, i32* %i, align 4
  %dec = add nsw i32 %1, -1
  store i32 %dec, i32* %i, align 4
  %2 = load i32, i32* %i, align 4
  %cmp1 = icmp eq i32 %2, 3
  br i1 %cmp1, label %if.then, label %if.end
  
  ; CHECK: if.then:
  ; CHECK-NEXT: call void @loop_end()
if.then:
  br label %while.end

if.end:
  br label %while.cond

while.end:
  ret void
}