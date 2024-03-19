#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

class PrintClassesVisitor
    : public clang::RecursiveASTVisitor<PrintClassesVisitor> {
public:
  bool VisitCXXRecordDecl(clang::CXXRecordDecl *decl) {
    if (decl->isStruct() || decl->isClass()) {
      llvm::outs() << decl->getNameAsString() << "\n";
      for (auto field : decl->fields()) {
        llvm::outs() << "|_" << field->getNameAsString() << "\n";
      }
      llvm::outs() << "\n";
    }
    return true;
  }
};

class PrintClassesConsumer : public clang::ASTConsumer {
public:
  void HandleTranslationUnit(clang::ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  PrintClassesVisitor Visitor;
};

class PrintClassesPlugin : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<PrintClassesConsumer>();
  }

protected:
  bool ParseArgs(const clang::CompilerInstance &Compiler,
                 const std::vector<std::string> &args) override {
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<PrintClassesPlugin>
    X("print-classes", "Prints description of class.");