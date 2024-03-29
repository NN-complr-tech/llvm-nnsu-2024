#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

class InlineVisitor : public clang::RecursiveASTVisitor<InlineVisitor> {
private:
  clang::ASTContext *Context;
  bool HasStatement;

public:
  explicit InlineVisitor(clang::ASTContext *Context)
      : Context(Context), HasStatement(false) {}

  bool VisitFunc(clang::FunctionDecl *Func) {
    HasStatement = false;
    clang::Stmt *BodyFunc = Func->getBody();

    for (const clang::Stmt *statements : BodyFunc->children()) {
      if (clang::isa<clang::IfStmt>(statements) ||
          clang::isa<clang::WhileStmt>(statements) ||
          clang::isa<clang::ForStmt>(statements) ||
          clang::isa<clang::DoStmt>(statements) ||
          clang::isa<clang::SwitchStmt>(statements)) {
        HasStatement = true;
      }
    }

    if (!HasStatement) {
      clang::SourceRange place = Func->getSourceRange();
      Func->addAttr(clang::AlwaysInlineAttr::CreateImplicit(*Context, place));
    }
    return true;
  }
};

class InlineConsumer : public clang::ASTConsumer {
private:
  InlineVisitor Visitor;

public:
  InlineConsumer(clang::ASTContext *Context) : Visitor(Context) {}

  void HandleTranslationUnit(clang::ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class InlinePlugin : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef) override {
    return std::make_unique<InlineConsumer>(&Compiler.getASTContext());
  }
  bool ParseArgs(const clang::CompilerInstance &Compiler,
                 const std::vector<std::string> &Args) override {
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<InlinePlugin>
    X("add-always-inline",
      "adds always_inline to functions without if statements.");
