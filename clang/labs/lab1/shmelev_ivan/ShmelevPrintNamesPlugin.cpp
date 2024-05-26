#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

class ClassExplorer : public clang::RecursiveASTVisitor<ClassExplorer> {
public:
  bool VisitCXXRecordDecl(clang::CXXRecordDecl *recordDecl) {
    if (!recordDecl->isClass() && !recordDecl->isStruct()) {
      return true;
    }

    outputClassName(recordDecl);
    outputClassMembers(recordDecl);
    llvm::outs() << "\n";
    return true;
  }

private:
  void outputClassName(const clang::CXXRecordDecl *recordDecl) const {
    llvm::outs() << recordDecl->getNameAsString() << "\n";
  }

  void outputClassMembers(const clang::CXXRecordDecl *recordDecl) const {
    for (const auto *member : recordDecl->fields()) {
      llvm::outs() << "|_" << member->getNameAsString() << "\n";
    }
  }
};

class ClassExplorerASTConsumer : public clang::ASTConsumer {
public:
  ClassExplorer visitor;

  void HandleTranslationUnit(clang::ASTContext &context) override {
    visitor.TraverseDecl(context.getTranslationUnitDecl());
  }
};

class ClassExplorerPluginAction : public clang::PluginASTAction {
public:
  bool displayHelp = false;

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &compiler,
                    llvm::StringRef) override {
    if (displayHelp) {
      return nullptr;
    }
    return std::make_unique<ClassExplorerASTConsumer>();
  }

  bool ParseArgs(const clang::CompilerInstance &compiler,
                 const std::vector<std::string> &args) override {
    for (const auto &arg : args) {
      if (arg == "--help") {
        llvm::outs() << "This plugin traverses the AST of the codebase and "
                        "prints the name and fields of each class or struct\n";
        displayHelp = true;
        break;
      }
    }
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<ClassExplorerPluginAction>
    registryEntry("classexplorer", "Prints all members of classes and structs");
