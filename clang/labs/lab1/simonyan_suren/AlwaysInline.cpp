#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/ADT/StringRef.h"
#include <stack>

class AlwaysInlineConsumer : public clang::ASTConsumer {
public:
  bool HandleTopLevelDecl(clang::DeclGroupRef DeclGroup) override {
    for (clang::Decl *Func : DeclGroup) {
      if (clang::isa<clang::FunctionDecl>(Func)) {
        if (Func->getAttr<clang::AlwaysInlineAttr>()) {
          continue;
        }
        clang::Stmt *Body = Func->getBody();
        if (Body != nullptr) {
          bool CondFound = false;
          std::stack<clang::Stmt *> Stack;
          Stack.push(Body);
          while (!Stack.empty() && !CondFound) {
            clang::Stmt *St = Stack.top();
            Stack.pop();
            for (clang::Stmt *StCh : St->children()) {
              if (clang::isa<clang::IfStmt>(St) ||
                  clang::isa<clang::WhileStmt>(St) ||
                  clang::isa<clang::ForStmt>(St) ||
                  clang::isa<clang::DoStmt>(St) ||
                  clang::isa<clang::SwitchStmt>(St)) {
                CondFound = true;
                break;
              }
              Stack.push(StCh);
            }
          }
          if (!CondFound) {
            clang::SourceLocation Location(Func->getSourceRange().getBegin());
            clang::SourceRange Range(Location);
            Func->addAttr(
                clang::AlwaysInlineAttr::Create(Func->getASTContext(), Range));
          }
        }
      }
    }
    return true;
  }
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
    X("always-inlines-plugin",
      "Print a function without conditions with an attribute");