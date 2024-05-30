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

namespace {
class AlwaysInlineVisitor
    : public clang::RecursiveASTVisitor<AlwaysInlineVisitor> {
public:
  bool VisitStmt(clang::Stmt *Statement) {
    if (clang::isa<clang::IfStmt>(Statement) ||
        clang::isa<clang::WhileStmt>(Statement) ||
        clang::isa<clang::ForStmt>(Statement) ||
        clang::isa<clang::DoStmt>(Statement) ||
        clang::isa<clang::SwitchStmt>(Statement)) {
      hasCondition = true;
    }
    return true;
  }

  bool hasCondition = false;
};

class AlwaysInlineConsumer : public clang::ASTConsumer {
public:
  bool HandleTopLevelDecl(clang::DeclGroupRef DeclGroup) override {
    for (clang::Decl *Decl : DeclGroup) {
      if (auto FuncDecl = clang::dyn_cast<clang::FunctionDecl>(Decl)) {
        if (FuncDecl->getAttr<clang::AlwaysInlineAttr>()) {
          continue;
        }
        clang::Stmt *Body = FuncDecl->getBody();
        if (Body) {
          AlwaysInlineVisitor Visitor;
          Visitor.TraverseStmt(Body);
          if (!Visitor.hasCondition) {
            clang::SourceLocation Location(
                FuncDecl->getSourceRange().getBegin());
            clang::SourceRange Range(Location);
            FuncDecl->addAttr(clang::AlwaysInlineAttr::Create(
                FuncDecl->getASTContext(), Range));
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
                 const std::vector<std::string> &args) override {
    for (const std::string &arg : args) {
      if (arg == "--help") {
        llvm::outs() << "adds always_inline if function has no conditions\n";
        return false;
      }
    }
    return true;
  }
};

} // namespace

static clang::FrontendPluginRegistry::Add<AlwaysInlinePlugin>
    X("bozin-always-inline",
      "adds always_inline if function has no conditions");