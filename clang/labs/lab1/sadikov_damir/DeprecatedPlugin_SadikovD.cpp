#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

class DeprecatedPluginVisitor
    : public clang::RecursiveASTVisitor<DeprecatedPluginVisitor> {
private:
  clang::ASTContext *Context;

public:
  explicit DeprecatedPluginVisitor(clang::CompilerInstance &Compiler)
      : Context(&Compiler.getASTContext()) {}

  bool VisitFunctionDecl(clang::FunctionDecl *FunDec) {
    std::string FunctionName = FunDec->getNameInfo().getAsString();
    if (FunctionName.find("deprecated") != std::string::npos) {
      clang::DiagnosticsEngine &Diags = Context->getDiagnostics();
      unsigned DiagID =
          Diags.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                                "Function name contains 'deprecated'");
      Diags.Report(FunDec->getLocation(), DiagID) << FunctionName;
    }
    return true;
  }
};

class DeprecatedPluginConsumer : public clang::ASTConsumer {
private:
  DeprecatedPluginVisitor Visitor;

public:
  explicit DeprecatedPluginConsumer(clang::CompilerInstance &Compiler)
      : Visitor(Compiler) {}
  void HandleTranslationUnit(clang::ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class DeprecatedPluginAction : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<DeprecatedPluginConsumer>(Compiler);
  }

protected:
  bool ParseArgs(const clang::CompilerInstance &Compiler,
                 const std::vector<std::string> &args) override {
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<DeprecatedPluginAction>
    X("DeprecatedPlugin_SadikovD",
      "Warning if function name contains 'deprecated'");
