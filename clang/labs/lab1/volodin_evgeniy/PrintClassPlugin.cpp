#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

class PrintClassVisitor : public RecursiveASTVisitor<PrintClassVisitor> {
public:
  explicit PrintClassVisitor(ASTContext *Context) : Context(Context) {}

  bool VisitCXXRecordDecl(CXXRecordDecl *declaration) {
    llvm::outs() << declaration->getNameAsString().c_str() << "\n";
    for (const auto &field : declaration->fields()) {
      llvm::outs() << " |_" << field->getNameAsString().c_str() << "\n";
    }
    return true;
  }

private:
  ASTContext *Context;
};

class PrintClassConsumer : public ASTConsumer {
public:
  explicit PrintClassConsumer(CompilerInstance &CI)
      : Visitor(&CI.getASTContext()) {}

  virtual void HandleTranslationUnit(ASTContext &Context) {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  PrintClassVisitor Visitor;
};

class PrintClassASTAction : public PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::unique_ptr<clang::ASTConsumer>(
        new PrintClassConsumer(Compiler));
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) {
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<PrintClassASTAction>
    X("print-class-plugin",
      "A plugin that prints the names of classes (structures), as well as the "
      "fields contained in them.");