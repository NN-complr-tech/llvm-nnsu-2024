// Copyright 2024 Sharapov Georgiy

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

class MyVisitor : public clang::RecursiveASTVisitor<MyVisitor> {
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

class MyConsumer : public clang::ASTConsumer {
 public:
    void HandleTranslationUnit(clang::ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }
 private:
    MyVisitor Visitor;
};

class MyAction : public clang::PluginASTAction {
 public:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
        clang::CompilerInstance &Compiler, llvm::StringRef InFile) override {
        return std::make_unique<MyConsumer>();
    }
 protected:
    bool ParseArgs(const clang::CompilerInstance &Compiler, const std::vector<std::string> &args) override {
        return true;
    }
};

static clang::FrontendPluginRegistry::Add<MyAction> X("myPlugin", "myPlugin");
