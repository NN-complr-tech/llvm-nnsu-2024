; RUN: opt -passes="bozin-inline" -S %s | FileCheck %s

define void @func_with_instr() {
  %1 = add i32 1, 2
  %2 = mul i32 %1, 3
  ret void
}

define void @caller_with_instr() {
  call void @func_with_instr()
  ret void
}

; CHECK-LABEL: @caller_with_instr
; CHECK-NOT: call void @func_with_instr

define void @func_with_args(i32 %x) {
  %1 = add i32 %x, 42
  ret void
}

define i32 @caller_with_args() {
  call void @func_with_args(i32 10)
  %result = add i32 5, 7
  ret i32 %result
}

; CHECK-LABEL: @caller_with_args
; CHECK: call void @func_with_args(i32 10)

define void @func_with_return() {
  %1 = add i32 1, 2
  ret void
}

define void @caller_with_return() {
  call void @func_with_return()
  ret void
}

; CHECK-LABEL: @caller_with_return
; CHECK-NOT: call void @func_with_return
; CHECK: %1 = add i32 1, 2

