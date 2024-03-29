#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

class AddWarningConsumer : public ASTConsumer {
  CompilerInstance &Instance;

public:
  AddWarningConsumer(CompilerInstance &Instance) : Instance(Instance) {}

  void HandleTranslationUnit(ASTContext &context) override {
    struct Visitor : public RecursiveASTVisitor<Visitor> {
      ASTContext *context;

      Visitor(ASTContext *context) : context(context) {}

      bool VisitFunctionDecl(FunctionDecl *FD) {
        std::string name = FD->getNameInfo().getAsString();
        if (name.find("deprecated") != -1) {
          DiagnosticsEngine &diag = context->getDiagnostics();
          unsigned diagID = diag.getCustomDiagID(
              DiagnosticsEngine::Warning, "Deprecated is contain in function name");
          SourceLocation location = FD->getLocation();
          diag.Report(location, diagID);
        }
        return true;
      }
    } v(&Instance.getASTContext());

    v.TraverseDecl(context.getTranslationUnitDecl());
   
  }
};

class AddWarningAction : public PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef InFile) override {
    return std::make_unique<AddWarningConsumer>(CI);
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    return true;
  }
};

static FrontendPluginRegistry::Add<AddWarningAction> X("warn_dep", "warn_dep");