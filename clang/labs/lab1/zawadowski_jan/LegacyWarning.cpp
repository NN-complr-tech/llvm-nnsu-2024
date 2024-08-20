#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include <cctype>

using namespace clang;

struct LegacyCodeVisitor : public RecursiveASTVisitor<LegacyCodeVisitor> {
  ASTContext *astContext;
  bool ignoreClassMembers;
  LegacyCodeVisitor(ASTContext *astContext, bool ignoreClassMembers)
      : astContext(astContext), ignoreClassMembers(ignoreClassMembers) {}
  bool VisitFunctionDecl(FunctionDecl *functionDecl) {
    if (!ignoreClassMembers || !functionDecl->isCXXClassMember()) {
      std::string string = functionDecl->getNameInfo().getAsString();
      std::transform(string.begin(), string.end(), string.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (string.find("legacy") != std::string::npos ||
          string.find("deprecated") != std::string::npos ||
          string.find("obsolete") != std::string::npos) {
        DiagnosticsEngine &diagnosticsEngine = astContext->getDiagnostics();
        diagnosticsEngine.Report(
            functionDecl->getLocation(),
            diagnosticsEngine.getCustomDiagID(
                DiagnosticsEngine::Warning,
                "Found potential legacy code usage: function or method "
                "contains special word in its name."))
            << string;
      }
    }
    return true;
  }
};

class LegacyCodeDetector : public ASTConsumer {
private:
  CompilerInstance &instance;
  bool ignoreClassMembers;

public:
  LegacyCodeDetector(CompilerInstance &instance, bool ignoreClassMembers)
      : instance(instance), ignoreClassMembers(ignoreClassMembers) {}
  void HandleTranslationUnit(ASTContext &context) override {
    LegacyCodeVisitor legacyCodeVisitor(&instance.getASTContext(),
                                        ignoreClassMembers);
    legacyCodeVisitor.TraverseDecl(context.getTranslationUnitDecl());
  }
};

class LegacyCodePluginAction : public PluginASTAction {
protected:
  bool ignoreClassMembers = false;
  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &compilerInstance,
                    llvm::StringRef inFile) override {
    return std::make_unique<LegacyCodeDetector>(compilerInstance,
                                                ignoreClassMembers);
  }
  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    for (const auto &arg : args) {
      if (arg == "-ignoreClassMembers") {
        ignoreClassMembers = true;
      }
    }
    return true;
  }
};

static FrontendPluginRegistry::Add<LegacyCodePluginAction>
    X("legacy-code", "Detect potential legacy code usage");
