#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::ast_matchers;

class ClassFieldPrinter : public MatchFinder::MatchCallback {
public:
  virtual void run(const MatchFinder::MatchResult &Result) {
    if (const CXXRecordDecl *ClassDecl =
            Result.Nodes.getNodeAs<CXXRecordDecl>("class")) {
      llvm::outs() << ClassDecl->getNameAsString() << "\n";
      for (const auto Field : ClassDecl->fields()) {
        llvm::outs() << "  |_ " << Field->getNameAsString() << "\n";
      }
    }
  }
};

class ClassFieldPrinterAction : public PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                  llvm::StringRef) override {
    Finder.addMatcher(
        cxxRecordDecl(hasDefinition(), unless(isImplicit())).bind("class"),
        &Printer);
    return std::make_unique<ASTConsumer>();
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    return true;
  }

private:
  MatchFinder Finder;
  ClassFieldPrinter Printer;
};

static FrontendPluginRegistry::Add<ClassFieldPrinterAction>
    X("class-field-printer", "print class names and fields");

