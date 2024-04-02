; RUN: opt -load-pass-plugin=%llvmshlibdir/AlexseevLoopPlugin%shlibext -passes=alexseev-loop-plugin -S %s | FileCheck %s

; void forFoo() {
;     int i = 5;
;     for (int i = 0; i < 5; i++) {
;         i++;
;     }
; }

; void whileFoo() {
;     int i = 10;
;     while (i > 0) {
;         i--;
;         if (i == 3)
;             break;
;     }
; }

; void foo() {
;     if (1) {
;         return;
;     }
; }

; CHECK-NOT: call void @loop_start()
; CHECK-NOT: call void @loop_end()

define dso_local void @foo() {
entry:
  br label %if.then

if.then:
  ret void
}
