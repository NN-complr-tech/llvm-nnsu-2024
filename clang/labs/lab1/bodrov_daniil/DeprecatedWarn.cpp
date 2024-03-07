#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/CompilerInstance.h"

using namespace clang;

class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor> {
public:
  explicit MyASTVisitor(ASTContext *Context) : Context(Context) {}

  bool VisitCXXRecordDecl(CXXRecordDecl *Declaration) {
    // Print the names of all classes and their fields
    if (Declaration->isThisDeclarationADefinition()) {
      llvm::outs() << Declaration->getNameAsString() << "\n";
      for (auto *Field : Declaration->fields()) {
        llvm::outs() << "  |_ " << Field->getNameAsString() << "\n";
      }
    }
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *F) {
    // Add always_inline attribute to all functions without conditions
    if (!F->hasBody() || !F->getBody()->containsConditionalStmt()) {
      F->addAttr(AlwaysInlineAttr::CreateImplicit(Context));
    }

    // Add warning if the function name contains "deprecated"
    if (F->getNameAsString().find("deprecated") != std::string::npos) {
      DiagnosticsEngine &D = Context->getDiagnostics();
      unsigned DiagID = D.getCustomDiagID(DiagnosticsEngine::Warning,
                                           "Function name contains 'deprecated'");
      D.Report(F->getLocation(), DiagID);
    }

    return true;
  }

private:
  ASTContext *Context;
};

class MyASTConsumer : public ASTConsumer {
public:
  explicit MyASTConsumer(ASTContext *Context) : Visitor(Context) {}

  virtual void HandleTranslationUnit(ASTContext &Context) {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  MyASTVisitor Visitor;
};

class MyPluginAction : public PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) {
    return std::make_unique<MyASTConsumer>(&CI.getASTContext());
  }

  bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string>& args) {
    return true;
  }
};

static FrontendPluginRegistry::Add<MyPluginAction> X("deprecated-warn", "Prints a warning if a function name contains 'deprecated'");
