#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

class DepVisitor : public RecursiveASTVisitor<DepVisitor> {
  ASTContext *Context;

public:
    explicit DepVisitor(ASTContext *Context) : Context(Context) {}

    bool VisitFunctionDecl(FunctionDecl *Func) {
        if (Func->getNameInfo().getAsString().find("deprecated") != std::string::npos) {
            DiagnosticsEngine &Diags = Context->getDiagnostics();
            unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Warning,
                                                   "Deprecated in function name");
            Diags.Report(Func->getLocation(), DiagID) << Func->getNameInfo().getAsString();
        }
        return true;
    }
};

class DepConsumer : public ASTConsumer {
  DepVisitor Visitor;

public:
    explicit DepConsumer(CompilerInstance &CI) : Visitor(&CI.getASTContext()) {}

    void HandleTranslationUnit(ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }
};

class DeprecatedPlugin : public PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer (
    CompilerInstance &Compiler, llvm::StringRef InFile) override {
      return std::unique_ptr<ASTConsumer>(new DepConsumer(Compiler));
    }

  bool ParseArgs(const CompilerInstance &Compiler, 
                 const std::vector<std::string> &args) override {
    return true;
  }
};

static FrontendPluginRegistry::Add<DeprecatedPlugin>
X("deprecated-match", "deprecated match");
