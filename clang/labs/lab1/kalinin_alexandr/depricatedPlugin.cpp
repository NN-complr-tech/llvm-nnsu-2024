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
