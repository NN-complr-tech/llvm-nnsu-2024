#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include <stack>

class AlwaysInlineVisitor
    : public clang::RecursiveASTVisitor<AlwaysInlineVisitor> {
public:
  AlwaysInlineVisitor(clang::ASTContext *MyContext) : MyContext(MyContext) {}

  bool VisitFunctionDecl(clang::FunctionDecl *Func) {
    bool containsConditional = false;
    std::stack<clang::Stmt *> stack;
    stack.push(Func->getBody());

    while (!stack.empty()) {
      clang::Stmt *currentNode = stack.top();
      stack.pop();

      if (clang::isa<clang::IfStmt>(currentNode) ||
          clang::isa<clang::SwitchStmt>(currentNode) ||
          clang::isa<clang::ForStmt>(currentNode) ||
          clang::isa<clang::WhileStmt>(currentNode)) {
        containsConditional = true;
        break;
      }

      if (auto parent = clang::dyn_cast<clang::CompoundStmt>(currentNode)) {
        for (auto Child : parent->body()) {
          stack.push(Child);
        }
      }
    }

    if (!containsConditional) {
      clang::SourceRange FuncRange = Func->getSourceRange();
      Func->addAttr(
          clang::AlwaysInlineAttr::CreateImplicit(*MyContext, FuncRange));
      auto thisAttr = Func->getAttr<clang::AlwaysInlineAttr>();
      llvm::outs() << "Added attribute " << thisAttr->getSpelling() << " in "
                   << Func->getNameAsString() << "\n";
    } else {
      llvm::outs() << Func->getNameAsString() << " "
                   << "not suitable for the attribute"
                   << "\n";
    }

    return true;
  }

private:
  clang::ASTContext *MyContext;
};

class AlwaysInlineConsumer : public clang::ASTConsumer {
public:
  AlwaysInlineConsumer(clang::ASTContext *MyContext) : MyVisitor(MyContext) {}

  void HandleTranslationUnit(clang::ASTContext &MyContext) override {
    MyVisitor.TraverseDecl(MyContext.getTranslationUnitDecl());
  }

private:
  AlwaysInlineVisitor MyVisitor;
};

class AlwaysInlinePlugin : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef) override {
    return std::make_unique<AlwaysInlineConsumer>(&Compiler.getASTContext());
  }
  bool ParseArgs(const clang::CompilerInstance &Compiler,
                 const std::vector<std::string> &Args) override {
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<AlwaysInlinePlugin>
    X("always_inlines-plugin",
      "Print a function without conditions with an attribute");