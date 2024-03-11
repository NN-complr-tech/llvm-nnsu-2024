#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

namespace {
class PrintClassMembersVisitor final
    : public clang::RecursiveASTVisitor<PrintClassMembersVisitor> {
public:
  explicit PrintClassMembersVisitor(clang::ASTContext *context)
      : m_context(context) {}
  bool VisitCXXRecordDecl(clang::CXXRecordDecl *declaration) {
    if (declaration->isStruct() || declaration->isClass()) {
      outInfoUserType(declaration);

      auto members = declaration->fields();

      for (const auto &member : members) {
        llvm::outs() << "|_ " << member->getNameAsString() << ' ';
        llvm::outs() << '(' << getTypeString(member) << '|';
        llvm::outs() << getAccessSpecifierAsString(member) << ")\n";
      }

      llvm::outs() << '\n';
    }
    return true;
  }

private:
  void outInfoUserType(clang::CXXRecordDecl *userType) {
    llvm::outs() << userType->getNameAsString() << ' ';
    llvm::outs() << (userType->isStruct() ? "(struct" : "(class");
    llvm::outs() << (userType->isTemplated() ? "|template)" : ")") << '\n';
  }

  std::string getTypeString(const clang::FieldDecl *member) {
    clang::QualType type = member->getType();
    return type.getAsString();
  }

  std::string getAccessSpecifierAsString(const clang::FieldDecl *member) {
    switch (member->getAccess()) {
    case clang::AS_public:
      return "public";
    case clang::AS_protected:
      return "protected";
    case clang::AS_private:
      return "private";
    default:
      return "unknown";
    }
  }

private:
  clang::ASTContext *m_context;
};

class PrintClassMembersConsumer final : public clang::ASTConsumer {
public:
  explicit PrintClassMembersConsumer(clang::ASTContext *сontext)
      : m_visitor(сontext) {}

  void HandleTranslationUnit(clang::ASTContext &context) override {
    m_visitor.TraverseDecl(context.getTranslationUnitDecl());
  }

private:
  PrintClassMembersVisitor m_visitor;
};

class PrintClassMembersAction final : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &ci, llvm::StringRef) override {
    return std::make_unique<PrintClassMembersConsumer>(&ci.getASTContext());
  }

  bool ParseArgs(const clang::CompilerInstance &ci,
                 const std::vector<std::string> &args) override {
    return true;
  }
};
} // namespace

static clang::FrontendPluginRegistry::Add<PrintClassMembersAction>
    X("pcm_plugin", "Prints all members of the class");