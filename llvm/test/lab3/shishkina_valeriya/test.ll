; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@ic = dso_local local_unnamed_addr global i64 0, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @_Z4funciii(i32 noundef %a, i32 noundef %b, i32 noundef %c) local_unnamed_addr #0 {
entry:
  store i64 0, ptr @ic, align 8, !tbaa !5
  %cmp5 = icmp sgt i32 %a, 0
  br i1 %cmp5, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %xtraiter = and i32 %a, 3
  %0 = icmp ult i32 %a, 4
  br i1 %0, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new

for.body.preheader.new:                           ; preds = %for.body.preheader
  %unroll_iter = and i32 %a, -4
  br label %for.body

for.cond.cleanup.loopexit.unr-lcssa:              ; preds = %for.body, %for.body.preheader
  %spec.select.lcssa.ph = phi i32 [ undef, %for.body.preheader ], [ %spec.select.3, %for.body ]
  %d.06.unr = phi i32 [ 0, %for.body.preheader ], [ %spec.select.3, %for.body ]
  %lcmp.mod.not = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod.not, label %for.cond.cleanup, label %for.body.epil

for.body.epil:                                    ; preds = %for.cond.cleanup.loopexit.unr-lcssa, %for.body.epil
  %d.06.epil = phi i32 [ %spec.select.epil, %for.body.epil ], [ %d.06.unr, %for.cond.cleanup.loopexit.unr-lcssa ]
  %epil.iter = phi i32 [ %epil.iter.next, %for.body.epil ], [ 0, %for.cond.cleanup.loopexit.unr-lcssa ]
  %cmp1.epil = icmp slt i32 %d.06.epil, %b
  %add.epil = select i1 %cmp1.epil, i32 %c, i32 0
  %spec.select.epil = add nsw i32 %add.epil, %d.06.epil
  %epil.iter.next = add i32 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i32 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %for.cond.cleanup, label %for.body.epil, !llvm.loop !9

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit.unr-lcssa, %for.body.epil, %entry
  %d.0.lcssa = phi i32 [ 0, %entry ], [ %spec.select.lcssa.ph, %for.cond.cleanup.loopexit.unr-lcssa ], [ %spec.select.epil, %for.body.epil ]
  ret i32 %d.0.lcssa

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %d.06 = phi i32 [ 0, %for.body.preheader.new ], [ %spec.select.3, %for.body ]
  %niter = phi i32 [ 0, %for.body.preheader.new ], [ %niter.next.3, %for.body ]
  %cmp1 = icmp slt i32 %d.06, %b
  %add = select i1 %cmp1, i32 %c, i32 0
  %spec.select = add nsw i32 %add, %d.06
  %cmp1.1 = icmp slt i32 %spec.select, %b
  %add.1 = select i1 %cmp1.1, i32 %c, i32 0
  %spec.select.1 = add nsw i32 %add.1, %spec.select
  %cmp1.2 = icmp slt i32 %spec.select.1, %b
  %add.2 = select i1 %cmp1.2, i32 %c, i32 0
  %spec.select.2 = add nsw i32 %add.2, %spec.select.1
  %cmp1.3 = icmp slt i32 %spec.select.2, %b
  %add.3 = select i1 %cmp1.3, i32 %c, i32 0
  %spec.select.3 = add nsw i32 %add.3, %spec.select.2
  %niter.next.3 = add i32 %niter, 4
  %niter.ncmp.3 = icmp eq i32 %niter.next.3, %unroll_iter
  br i1 %niter.ncmp.3, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body, !llvm.loop !11
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 17.0.6 (https://github.com/ShLera04/llvm-nnsu-2024.git 450b2a40cb91b60eaf1307df0cc91943fb7c5d74)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"long", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.disable"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
