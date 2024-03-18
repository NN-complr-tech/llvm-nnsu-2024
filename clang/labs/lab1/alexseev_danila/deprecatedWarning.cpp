#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

class DepFuncVisitor : public RecursiveASTVisitor<DepFuncVisitor> {
private:
  ASTContext *Context;
  std::string ExcludeFunc;

public:
  explicit DepFuncVisitor(ASTContext *Context) : Context(Context) {}

  bool VisitFunctionDecl(FunctionDecl *Func) {
	if (Func->getNameInfo().getAsString() == ExcludeFunc) {
      return true;
    }
    if (Func->getNameInfo().getAsString().find("deprecated") !=
        std::string::npos) {
      DiagnosticsEngine &Diags = Context->getDiagnostics();
      size_t CustomDiagID =
          Diags.getCustomDiagID(DiagnosticsEngine::Warning,
                                "Function contains 'deprecated' in its name");
      Diags.Report(Func->getLocation(), CustomDiagID)
          << Func->getNameInfo().getAsString();
    }
    return true;
  }
};

class DepFuncConsumer : public ASTConsumer {
public:
  explicit DepFuncConsumer(CompilerInstance &CI) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    DepFuncVisitor Visitor(&Context);
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
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
                 const std::vector<std::string> &Args) override {
    for (size_t i = 0; i < Args.size(); ++i) {
      const std::string &Arg = Args[i];
      if (Arg == "-excluding") {
        ExcludeFunc = Args[i + 1];
        i++;
      }
    }
    return true;
  }
};

static FrontendPluginRegistry::Add<DepFuncPlugin> X("deprecated-warning",
                                                    "deprecated warning");
