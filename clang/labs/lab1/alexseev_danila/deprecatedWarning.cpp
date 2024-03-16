#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

class DepFuncConsumer : public ASTConsumer {
  CompilerInstance &Instance;

public:
  explicit DepFuncConsumer(CompilerInstance &CI) : Instance(CI) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    struct DepFuncVisitor : public RecursiveASTVisitor<DepFuncVisitor> {
      ASTContext *Context;

      DepFuncVisitor(ASTContext *Context) : Context(Context) {}

      bool VisitFunctionDecl(FunctionDecl *Func) {
        if (Func->getNameInfo().getAsString().find("deprecated") !=
            std::string::npos) {
          DiagnosticsEngine &Diags = Context->getDiagnostics();
          unsigned DiagID = Diags.getCustomDiagID(
              DiagnosticsEngine::Warning, "Function contains 'deprecated' in its name");
          Diags.Report(Func->getLocation(), DiagID)
              << Func->getNameInfo().getAsString();
        }
        return true;
      }
    } visitor(&Instance.getASTContext());

    visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class DepFuncPlugin : public PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<DepFuncConsumer>(Compiler);
  }

  bool ParseArgs(const CompilerInstance &Compiler,
                 const std::vector<std::string> &args) override {
    return true;
  }
};

static FrontendPluginRegistry::Add<DepFuncPlugin> X("deprecated-warning",
                                                                "deprecated warning");
