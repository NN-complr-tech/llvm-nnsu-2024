#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

class DeprecatedFunctionVisitor : public RecursiveASTVisitor<DeprecatedFunctionVisitor> {
public:
  explicit DeprecatedFunctionVisitor(ASTContext *Context) : Context(Context) {}

  bool VisitFunctionDecl(FunctionDecl *Declaration) {
    if (Declaration->getNameAsString().find("deprecated") != std::string::npos) {
      llvm::outs() << "Warning: Function '" << Declaration->getNameAsString()
                   << "' is marked as deprecated\n";
    }

    return true;
  }

private:
  ASTContext *Context;
};

class DeprecatedFunctionConsumer : public ASTConsumer {
public:
  explicit DeprecatedFunctionConsumer(ASTContext *Context) : Visitor(Context) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  DeprecatedFunctionVisitor Visitor;
};

class DeprecatedFunctionAction : public PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef) override {
    return std::make_unique<DeprecatedFunctionConsumer>(&CI.getASTContext());
  }

  bool ParseArgs(const CompilerInstance &,
                 const std::vector<std::string> &) override {
    return true;
  }
};

static FrontendPluginRegistry::Add<DeprecatedFunctionAction>
    Y("deprecated-function-warning", "warn about functions marked as deprecated");
