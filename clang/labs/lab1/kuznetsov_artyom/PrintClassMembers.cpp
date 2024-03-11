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

class PrintClassMembersConsumer final : public ASTConsumer {
public:
  explicit PrintClassMembersConsumer(ASTContext *сontext)
      : m_visitor(сontext) {}

  void HandleTranslationUnit(ASTContext &context) override {
    m_visitor.TraverseDecl(context.getTranslationUnitDecl());
  }

private:
  PrintClassMembersVisitor m_visitor;
};

class PrintClassMembersAction final : public PluginASTAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &ci,
                                                 StringRef) override {
    return std::make_unique<PrintClassMembersConsumer>(&ci.getASTContext());
  }

  bool ParseArgs(const CompilerInstance &ci,
                 const std::vector<std::string> &args) override {
    return true;
  }
};
} // namespace

static FrontendPluginRegistry::Add<PrintClassMembersAction>
    X("pcm_plugin", "Prints all members of the class");