#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Basic/AttrKinds.h" // Include this header for AlwaysInlineAttr

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
    if (!F->hasBody() || !F->getBody() || !containsIfStmt(F->getBody())) {
      F->addAttr(AlwaysInlineAttr::CreateImplicit(*Context, SourceRange(), AlwaysInlineAttr::GNU_inline));
    }

    // Add warning if the function name contains "deprecated"
    if (F->getNameAsString().find("deprecated") != std::string::npos) {
      DiagnosticsEngine &D = Context->getDiagnostics();
      unsigned DiagID = D.getCustomDiagID(DiagnosticsEngine::Warning,
                                           "Function '%0' name contains 'deprecated'");
      D.Report(F->getLocation(), DiagID) << F->getNameAsString();
    }

    return true;
  }

  bool VisitVarDecl(VarDecl *VD) {
    // Renaming the variable
    if (VD->getNameAsString() == "oldName") {
      VD->setDeclName(Context->DeclarationNames.getIdentifier("newName"));
    }
    return true;
  }

  bool VisitNamedDecl(NamedDecl *ND) {
    // Renaming the class
    if (auto *CXX = dyn_cast<CXXRecordDecl>(ND)) {
      if (CXX->getNameAsString() == "OldClassName") {
        CXX->setDeclName(Context->DeclarationNames.getCXXRecordName("NewClassName"));
      }
    }
    return true;
  }

private:
  ASTContext *Context;

  bool containsIfStmt(Stmt *S) {
    if (!S)
      return false;
    if (isa<IfStmt>(S))
      return true;
    for (Stmt *Child : S->children()) {
      if (containsIfStmt(Child))
        return true;
    }
    return false;
  }
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
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) override {
    return std::make_unique<MyASTConsumer>(&CI.getASTContext());
  }

  bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string>& args) override {
    return true;
  }
};

static FrontendPluginRegistry::Add<MyPluginAction> X("deprecated-warn", "Prints a warning if a function name contains 'deprecated'");
