#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

class PrintClassesVisitor
    : public clang::RecursiveASTVisitor<PrintClassesVisitor> {
public:
  explicit PrintClassesVisitor(clang::ASTContext *Context) : Context(Context) {}

  bool VisitCXXRecordDecl(clang::CXXRecordDecl *Rst) {
    llvm::outs() << Rst->getNameAsString() << "\n";

    for (clang::Decl *dclrt : Rst->decls()) {
      if (auto *fld = clang::dyn_cast<clang::FieldDecl>(dclrt)) {
        llvm::outs() << "  |_ " << fld->getNameAsString() << "\n";
      } else if (auto *staticFld = clang::dyn_cast<clang::VarDecl>(dclrt)) {
        if (staticFld->isStaticDataMember()) {
          llvm::outs() << "  |_ " << staticFld->getNameAsString() << "\n";
        }
      }
    }
    return true;
  }

private:
  clang::ASTContext *Context;
};

class PrintClassesConsumer : public clang::ASTConsumer {
public:
  explicit PrintClassesConsumer(clang::ASTContext *Context)
      : Visitor(Context) {}

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
    return std::make_unique<PrintClassesConsumer>(&Compiler.getASTContext());
  }

protected:
  bool ParseArgs(const clang::CompilerInstance &Compiler,
                 const std::vector<std::string> &Args) override {
    for (const std::string &arg : Args) {
      if (arg == "--help") {
        llvm::outs() << "Display names of classes.\n";
      }
    }
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<PrintClassesPlugin>
    X("mirzakhmedov-classes-print", "Prints classes description.");