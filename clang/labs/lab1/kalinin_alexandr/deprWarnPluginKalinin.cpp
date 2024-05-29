#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

class CustomNodeVisitor : public RecursiveASTVisitor<CustomNodeVisitor> {
public:
  CustomNodeVisitor(bool caseSensitive) : CaseSensitive(caseSensitive) {}

  bool VisitFunctionDecl(FunctionDecl *fDecl) {
    if (fDecl && fDecl->isFunctionOrFunctionTemplate()) {
      std::string functionName = fDecl->getNameInfo().getAsString();
      std::string keyword = "deprecated";

      if (!CaseSensitive) {
        // Convert both strings to lower case for case-insensitive comparison
        std::transform(functionName.begin(), functionName.end(),
                       functionName.begin(), ::tolower);
        std::transform(keyword.begin(), keyword.end(), keyword.begin(),
                       ::tolower);
      }

      if (functionName.find(keyword) != std::string::npos) {
        DiagnosticsEngine &diagn = fDecl->getASTContext().getDiagnostics();
        unsigned diagnID = diagn.getCustomDiagID(
            DiagnosticsEngine::Warning, "The function name has 'deprecated'");
        diagn.Report(fDecl->getLocation(), diagnID)
            << fDecl->getNameInfo().getAsString();
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
        CI.getDiagnostics().Report(llvm::diag::warn_invalid_value)
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
