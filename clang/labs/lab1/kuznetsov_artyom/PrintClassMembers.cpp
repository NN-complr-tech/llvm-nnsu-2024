#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;
using namespace llvm;

namespace {
class PrintClassMembersVisitor final
    : public RecursiveASTVisitor<PrintClassMembersVisitor> {
public:
  explicit PrintClassMembersVisitor(ASTContext *context) : m_context(context) {}
  bool VisitCXXRecordDecl(CXXRecordDecl *declaration) {
    declaration->dump();
    return true;
  }

private:
  ASTContext *m_context;
};
}