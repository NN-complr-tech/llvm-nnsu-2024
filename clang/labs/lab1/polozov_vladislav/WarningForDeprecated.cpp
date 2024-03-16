#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace clang;

class WarningDeprecatedConsumer : public ASTConsumer {
  CompilerInstance &Instance;

public:
  WarningDeprecatedConsumer(CompilerInstance &Instance) : Instance(Instance) {}

  void HandleTranslationUnit(ASTContext &context) override {

    struct Visitor : public RecursiveASTVisitor<Visitor> {
      ASTContext *Context;
      Visitor(ASTContext *_Context) : Context(_Context) {}
      bool VisitFunctionDecl(FunctionDecl *FD) {
        std::string want_find = "deprecated";
        std::string name = FD->getNameInfo().getAsString();
        std::transform(name.begin(),name.end(),name.begin(), ::tolower);
        if (name.find(want_find) != std::string::npos) {
          DiagnosticsEngine &Diags = Context->getDiagnostics();
          unsigned DiagID = Diags.getCustomDiagID(
              DiagnosticsEngine::Warning, "find 'deprecated' in function name");
          Diags.Report(FD->getLocation(), DiagID)
              << FD->getNameInfo().getAsString();
        }
        return true;
      }
    };
    Visitor MyVisitor(&Instance.getASTContext());
    MyVisitor.TraverseDecl(context.getTranslationUnitDecl());
  }
};

class WarningDeprecatedAction : public PluginASTAction {

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef) override {
    return std::make_unique<WarningDeprecatedConsumer>(CI);
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    if (!args.empty() && args[0] == "help")
      PrintHelp(llvm::errs());
    return true;
  }
  void PrintHelp(llvm::raw_ostream &ros) {
    ros << "Plugin Warning Deprecated prints a warning if a function name contains 'deprecated'\n";
  }
};

static FrontendPluginRegistry::Add<WarningDeprecatedAction>
    X("warning-deprecated",
      "Prints a warning if a function name contains 'deprecated'");