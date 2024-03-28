#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

class AlwaysInlineVisitor : public RecursiveASTVisitor<AlwaysInlineVisitor> {
public:
  AlwaysInlineVisitor(ASTContext *Context) : Context(Context) {}

  bool VisitFunctionDecl(FunctionDecl *func) {
    Stmt *Body = func->getBody();
    if (Body && isa<CompoundStmt>(Body)) {
      CompoundStmt *BodyCompound = cast<CompoundStmt>(Body);
      for (Stmt *S : BodyCompound->body()) {
        if (!isa<IfStmt>(S)) {
          func->addAttr(AlwaysInlineAttr::CreateImplicit(*Context));
          llvm::outs() << "__attribute__((always_inline)) "
                       << func->getNameAsString() << "\n";
        }
      }
    }
    return true;
  }

private:
  ASTContext *Context;
};

class AlwaysInlineConsumer : public ASTConsumer {
public:
  AlwaysInlineConsumer(ASTContext *Context) : Visitor(Context) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  AlwaysInlineVisitor Visitor;
};

class AlwaysInlinePlugin : public PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(CompilerInstance &Compiler, llvm::StringRef) override {
    return std::make_unique<AlwaysInlineConsumer>(&Compiler.getASTContext());
  }
  bool ParseArgs(const CompilerInstance &Compiler,
                 const std::vector<std::string> &args) override {
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<AddAlwaysInlineAction>
X("always_inline", "Automatically adds attribute((always_inline)) "
"to functions without conditional statements.");