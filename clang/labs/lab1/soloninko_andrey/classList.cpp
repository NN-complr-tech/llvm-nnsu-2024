#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

class PrintClass : public RecursiveASTVisitor<PrintClass> {
public:
  bool VisitCXXRecordDecl(CXXRecordDecl *declaration) {
    if (declaration->isClass() || declaration->isStruct()) {
      llvm::outs() << declaration->getNameAsString() << "\n";
      for (auto field_member : declaration->fields()) {
        llvm::outs() << "|_" << field_member->getNameAsString() << "\n";
      }
      llvm::outs() << "\n";
    }
    return true;
  }
};

class MyASTConsumer : public ASTConsumer {
public:
  PrintClass Visitor;
  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class PrintClassAction : public PluginASTAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &ci,
                                                 llvm::StringRef) override {
    return std::make_unique<MyASTConsumer>();
  }

  bool ParseArgs(const CompilerInstance &ci,
                 const std::vector<std::string> &args) override {
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<PrintClassAction>
    X("class_list_plugin", "Prints all members of the class");