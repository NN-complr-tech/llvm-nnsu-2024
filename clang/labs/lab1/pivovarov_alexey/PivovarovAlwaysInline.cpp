#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include <queue>

class AlwaysInlineConsumer : public clang::ASTConsumer {
public:
  bool HandleTopLevelDecl(clang::DeclGroupRef DeclGroup) override {
    for (clang::Decl *Decl : DeclGroup) {
      if (clang::isa<clang::FunctionDecl>(Decl)) {
        if (Decl->getAttr<clang::AlwaysInlineAttr>()) {
          continue;
        }
        clang::Stmt *Body = Decl->getBody();
        if (Body != nullptr) {
          bool CondFound = false;
          std::queue<clang::Stmt *> StQueue;
          StQueue.push(Body);
          while (!StQueue.empty() && !CondFound) {
            clang::Stmt *St = StQueue.front();
            StQueue.pop();
            if (clang::isa<clang::IfStmt>(St) ||
                clang::isa<clang::WhileStmt>(St) ||
                clang::isa<clang::ForStmt>(St) ||
                clang::isa<clang::DoStmt>(St) ||
                clang::isa<clang::SwitchStmt>(St)) {
              CondFound = true;
              break;
            }
            for (clang::Stmt *StCh : St->children()) {
              StQueue.push(StCh);
            }
          }
          if (!CondFound) {
            clang::SourceLocation Location(Decl->getSourceRange().getBegin());
            clang::SourceRange Range(Location);
            Decl->addAttr(
                clang::AlwaysInlineAttr::Create(Decl->getASTContext(), Range));
          }
        }
      }
    }
    return true;
  }
};

class AlwaysInlinePlugin : public clang::PluginASTAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<AlwaysInlineConsumer>();
  }

  bool ParseArgs(const clang::CompilerInstance &Compiler,
                 const std::vector<std::string> &Args) override {
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<AlwaysInlinePlugin>
    X("always-inline-plugin",
      "Print a function without conditions with an attribute");
