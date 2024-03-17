#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

class DepFuncVisitor : public RecursiveASTVisitor<DepFuncVisitor> {
private:
    ASTContext *Context;

public:
    explicit DepFuncVisitor(ASTContext *context) : Context(context) {}

    bool VisitFunctionDecl(FunctionDecl *func) {
        if (func->getNameInfo().getAsString().find("deprecated") !=
            std::string::npos) {
            DiagnosticsEngine &diags = Context->getDiagnostics();
            size_t customDiagID = diags.getCustomDiagID(
                DiagnosticsEngine::Warning, "Function contains 'deprecated' in its name");
            diags.Report(func->getLocation(), customDiagID)
                << func->getNameInfo().getAsString();
        }
        return true;
    }
};

class DepFuncConsumer : public ASTConsumer {
private:
    CompilerInstance &instance;

public:
    explicit DepFuncConsumer(CompilerInstance &CI) : instance(CI) {}

    void HandleTranslationUnit(ASTContext &context) override {
        DepFuncVisitor visitor(&instance.getASTContext());
        visitor.TraverseDecl(context.getTranslationUnitDecl());
    }
};


class DepFuncPlugin : public PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer>  CreateASTConsumer(CompilerInstance &compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<DepFuncConsumer>(compiler);
  }

  bool ParseArgs(const CompilerInstance &compiler,
                 const std::vector<std::string> &args) override {
    return true;
  }
};

static FrontendPluginRegistry::Add<DepFuncPlugin> X("deprecated-warning",
                                                                "deprecated warning");
