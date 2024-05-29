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
    std::string NameOfFunction = fDecl->getNameInfo().getAsString();
    if (!CaseSensitive) {
      std::transform(NameOfFunction.begin(), NameOfFunction.end(),
                     NameOfFunction.begin(), ::tolower);
    }
    if (NameOfFunction.find("deprecated") != std::string::npos) {
      DiagnosticsEngine &Diagnostics = fDecl->getASTContext().getDiagnostics();
      unsigned int DiagnosticsId = Diagnostics.getCustomDiagID(
          DiagnosticsEngine::Warning,
          "The function name contains \"deprecated\"");
      SourceLocation PositionOfFunction = fDecl->getLocation();
      Diagnostics.Report(PositionOfFunction, DiagnosticsId) << NameOfFunction;
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
      } else {
        CaseSensitive = true;
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
