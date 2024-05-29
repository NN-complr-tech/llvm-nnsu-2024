#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

class CustomNodeVisitor : public RecursiveASTVisitor<CustomNodeVisitor> {
public:
  CustomNodeVisitor(bool caseSensitive) : CaseSensitive(caseSensitive) {}

  bool VisitFunctionDecl(FunctionDecl *fDecl) {
    if (fDecl && fDecl->isFunctionOrFunctionTemplate()) {
      StringRef functionName = fDecl->getNameInfo().getAsString();
      StringRef keyword = "deprecated";

      if (!CaseSensitive) {
        // Perform case-insensitive comparison
        if (functionName.lower() == keyword.lower()) {
          DiagnosticsEngine &diagn = fDecl->getASTContext().getDiagnostics();
          unsigned diagnID = diagn.getCustomDiagID(
              DiagnosticsEngine::Warning, "The function name has 'deprecated'");
          diagn.Report(fDecl->getLocation(), diagnID) << functionName;
        }
      } else {
        // Case-sensitive comparison
        if (functionName == keyword) {
          DiagnosticsEngine &diagn = fDecl->getASTContext().getDiagnostics();
          unsigned diagnID = diagn.getCustomDiagID(
              DiagnosticsEngine::Warning, "The function name has 'deprecated'");
          diagn.Report(fDecl->getLocation(), diagnID) << functionName;
        }
      }
    }
    return true;
  }

private:
  bool CaseSensitive;
};

class CustomConsumer : public ASTConsumer {
public:
  CustomConsumer(bool caseSensitive) : CaseSensitive(caseSensitive) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    CustomNodeVisitor Cnv(CaseSensitive);
    Cnv.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  bool CaseSensitive;
};

class PluginDeprFunc : public PluginASTAction {
public:
  PluginDeprFunc() : CaseSensitive(true) {}

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &Instance,
                    llvm::StringRef InFile) override {
    return std::make_unique<CustomConsumer>(CaseSensitive);
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &Args) override {
    for (const auto &Arg : Args) {
      if (Arg == "-case-insensitive") {
        CaseSensitive = false;
      } else if (Arg == "-case-sensitive") {
        CaseSensitive = true;
      } else {
        CI.getDiagnostics().Report(diag::err_drv_invalid_value)
            << Arg << "invalid argument";
      }
    }
    return true;
  }

private:
  bool CaseSensitive;
};

static FrontendPluginRegistry::Add<PluginDeprFunc>
    X("plugin_for_deprecated_functions",
      "If the function name contains \"deprecated\" plugin writes a warning");
