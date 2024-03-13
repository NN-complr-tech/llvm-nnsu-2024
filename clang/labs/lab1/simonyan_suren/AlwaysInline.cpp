#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

class AlwaysInlineVisitor : public RecursiveASTVisitor<AlwaysInlineVisitor> {
  private:
  ASTContext *myContext;

  public:
  AlwaysInlineVisitor(ASTContext *myContext) : myContext(myContext) {}

  bool VisitFunctionDecl(FunctionDecl *func) {
    
    if (func->isInlined()) {
      return true; // Function is already inline, no need to add the attribute
    }
    Stmt *Body = func->getBody();
      if (Body && isa<CompoundStmt>(Body)) {
      CompoundStmt *BodyCompound = cast<CompoundStmt>(Body);
      bool hasConditions = false;
      for (Stmt *S : BodyCompound->body()) {
        if (isa<IfStmt>(S)) {
          hasConditions = true;
          break;
        }
      }
      if (!hasConditions) {
        func->addAttr(AlwaysInlineAttr::CreateImplicit(*myContext));
        llvm::outs() << "__attribute__((always_inline)) "
        << func->getNameAsString() << "\n";
      }
    }
    return true;
  }
  };

class AlwaysInlineConsumer : public ASTConsumer {
public:
  AlwaysInlineConsumer(ASTContext *myContext) : myVisitor(myContext) {}

  void HandleTranslationUnit(ASTContext &myContext) override {
    myVisitor.TraverseDecl(myContext.getTranslationUnitDecl());
  }

private:
  AlwaysInlineVisitor myVisitor;
};

class AlwaysInlinePlugin : public PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(CompilerInstance &Compiler,
                    llvm::StringRef) override {
    return std::make_unique<AlwaysInlineConsumer>(&Compiler.getASTContext());
  }
  bool ParseArgs(const CompilerInstance &Compiler,
                 const std::vector<std::string> &args) override {
    return true;
  }
};

static FrontendPluginRegistry::Add<AlwaysInlinePlugin>
    X("always-inlines-plugin", "Print a function without conditions with an attribute");