#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

class CustomNodeVisitor : public RecursiveASTVisitor<CustomNodeVisitor> {
  bool CaseInsensitive;

public:
  CustomNodeVisitor(bool CaseInsensitive) : CaseInsensitive(CaseInsensitive) {}
  bool VisitFunctionDecl(FunctionDecl *Pfunction) {
    if (fDecl->getNameInfo().getAsString().find("deprecated") != std::string::npos) {
      DiagnosticsEngine &diagn = fDecl->getASTContext().getDiagnostics();
      unsigned diagnID = diagn.getCustomDiagID(DiagnosticsEngine::Warning, "The function name has 'deprecated'");
      diagn.Report(fDecl->getLocation(), diagnID) << fDecl->getNameInfo().getAsString();
    }
    return true;
  }
};

class CustomConsumer : public ASTConsumer {
  bool CaseInsensitive;

public:
  explicit CustomConsumer(bool CaseInsensitive)
      : CaseInsensitive(CaseInsensitive) {}
  void HandleTranslationUnit(ASTContext &Context) override {
    CustomNodeVisitor Cnv(CaseInsensitive);
    Cnv.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class PluginDeprFunc : public PluginASTAction {
  bool CaseInsensitive = false;
  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &Instance,
                    llvm::StringRef InFile) override {
    return std::make_unique<CustomConsumer>(CaseInsensitive);
  }
  bool ParseArgs(const CompilerInstance &Compiler,
                 const std::vector<std::string> &Args) override {
    for (const auto &arg : Args) {
      if (arg == "-i") {
        CaseInsensitive = true;
      }
    }
    return true;
  }
};

static FrontendPluginRegistry::Add<PluginDeprFunc>
    X("plugin_for_deprecated_functions",
      "If the function name contains \"deprecated\" plugin writes a warning");
