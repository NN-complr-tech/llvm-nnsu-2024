#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

class DepWarningVisitor : public clang::RecursiveASTVisitor<DepWarningVisitor> {
 public:
    bool VisitFunctionDecl(clang::FunctionDecl *F) {
        if (F->hasAttr<clang::DeprecatedAttr>()) {
            clang::DiagnosticsEngine &Diag = F->getASTContext().getDiagnostics();
            unsigned DiagID = Diag.getCustomDiagID(clang::DiagnosticsEngine::Warning, "function '%0' is deprecated");
            Diag.Report(F->getLocation(), DiagID) << F->getNameAsString();
        }
        return true;
    }
};

class DepWarningConsumer : public clang::ASTConsumer {
 public:
    void HandleTranslationUnit(clang::ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }
 private:
    DepWarningVisitor Visitor;
};

class DepWarningAction : public clang::PluginASTAction {
 public:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
        clang::CompilerInstance &Compiler, llvm::StringRef InFile) override {
        return std::make_unique<DepWarningConsumer>();
    }
 protected:
    bool ParseArgs(const clang::CompilerInstance &Compiler, const std::vector<std::string> &args) override {
        return true;
    }
};

static clang::FrontendPluginRegistry::Add<DepWarningAction>
X("DepEmitWarning", "Emit warning for each deprecated function");
