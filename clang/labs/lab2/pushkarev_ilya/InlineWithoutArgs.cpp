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

class InlineWithoutArgsVisitor
    : public clang::RecursiveASTVisitor<InlineWithoutArgsVisitor> {
public:
  bool VisitCallExpr(clang::CallExpr *Call) {

    // Check if function call: a) doesn't have arguments, b) is void c) isn't a
    // class method

    if (Call->getNumArgs() == 0 && !Call->getType()->isVoidType() &&
        !isa<clang::CXXMemberCallExpr>(Call)) {
      clang::FunctionDecl *CalleeDecl = Call->getDirectCallee();
      if (CalleeDecl && !CalleeDecl->hasBody()) {
        // insert function body
        clang::SourceLocation Loc = Call->getLocStart();
        Call->getASTContext().getLangOpts().InsertPragmaInline(
            Loc, CalleeDecl->getBody(), CalleeDecl->getLocStart(),
            CalleeDecl->getLocEnd());
        // deleting function call
        Call->RemoveFromParent();
      }
    }
    return true;
  }
};

class InlineWithoutArgsConsumer : public clang::ASTConsumer {
public:
  bool HandleTopLevelDecl(clang::DeclGroupRef DeclGroup) override {
    for (clang::Decl *Decl : DeclGroup) {
      if (auto FuncDecl = clang::dyn_cast<clang::FunctionDecl>(Decl)) {
        // check if void and zero args
        if (FuncDecl->getNumParams() == 0 &&
            FuncDecl->getReturnType()->isVoidType()) {
          clang::Stmt *Body = FuncDecl->getBody();
          if (Body != nullptr) {
            InlineWithoutArgsVisitor Visitor;
            Visitor.TraverseStmt(Body);
          }
        }
      }
    }
    return true;
  }
};

class InlineWithoutArgsPlugin : public clang::PluginASTAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<InlineWithoutArgsConsumer>();
  }

  bool ParseArgs(const clang::CompilerInstance &Compiler,
                 const std::vector<std::string> &Args) override {
    for (const std::string &Arg : Args) {
      if (Arg == "--help") {
        llvm::outs() << "inserts function body instead of call when return "
                        "type is void and no arguments required\n";
        return false;
      }
    }
    return true;
  }
};

} // namespace

static clang::FrontendPluginRegistry::Add<InlineWithoutArgsPlugin>
    X("inline-without-args", "inserts function body instead of call when "
                             "return type is void and no arguments required");
