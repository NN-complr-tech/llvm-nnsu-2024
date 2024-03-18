#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

class PrintClassVisitor : public clang::RecursiveASTVisitor<PrintClassVisitor> {
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

class PrintClassConsumer : public clang::ASTConsumer {
public:
  void HandleTranslationUnit(clang::ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  PrintClassVisitor Visitor;
};

class PrintClassPlugin : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer (
    clang::CompilerInstance &Compiler, llvm::StringRef InFile) override {
      return std::make_unique<PrintClassConsumer>();
    }

protected:
  bool ParseArgs(const clang::CompilerInstance &Compiler, 
                 const std::vector<std::string> &args) override {
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<PrintClassPlugin>
  X("print-class", "Prints description of class.");
  