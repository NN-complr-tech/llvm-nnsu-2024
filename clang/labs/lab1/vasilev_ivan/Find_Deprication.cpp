#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

class DeprecationWarnConsumer : public clang::ASTConsumer {
public:
  void HandleTranslationUnit(clang::ASTContext &Context) override {
    struct DeprecationWarnVisitor
        : public clang::RecursiveASTVisitor<DeprecationWarnVisitor> {
      clang::ASTContext &Context;

    public:
      explicit DeprecationWarnVisitor(clang::ASTContext &Context)
          : Context(Context) {}

      bool VisitFunctionDecl(clang::FunctionDecl *Func) {
        if (Func->getNameAsString().find("deprecated") != std::string::npos) {
          clang::DiagnosticsEngine &Diags = Context.getDiagnostics();
          unsigned DiagID =
              Diags.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                                    "Deprecated function found here");
          Diags.Report(Func->getLocation(), DiagID);
        }
        return true;
      }
    } Visitor(Context);
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class DeprecationWarnPlugin : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<DeprecationWarnConsumer>();
  }

protected:
  bool ParseArgs(const clang::CompilerInstance &Compiler,
                 const std::vector<std::string> &args) override {
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<DeprecationWarnPlugin>
    X("deprecation-plugin", "Finds deprecated functions");