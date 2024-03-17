#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

namespace {

class MemberInfoPrinter {
public:
  void print(const clang::ValueDecl *member, const std::string &memberType) {
    llvm::outs() << "|_ " << member->getNameAsString() << ' ';
    llvm::outs() << '(' << member->getType().getAsString() << '|';
    llvm::outs() << getAccessSpecifierAsString(member) << (memberType == "field" ? ")" : ("|" + memberType + ")") ) << "\n";
  }

private:
  std::string getAccessSpecifierAsString(const clang::ValueDecl *member) {
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
};

class UserTypePrinter {
public:
  void print(clang::CXXRecordDecl *userType) {
    llvm::outs() << userType->getNameAsString() << ' ';
    llvm::outs() << (userType->isStruct() ? "(struct" : "(class");
    llvm::outs() << (userType->isTemplated() ? "|template)" : ")") << '\n';
  }
};

class ClassMembersPrinter final : public clang::RecursiveASTVisitor<ClassMembersPrinter> {
public:
  explicit ClassMembersPrinter(clang::ASTContext *context) : context_(context) {}

  bool VisitCXXRecordDecl(clang::CXXRecordDecl *declaration) {
    if (declaration->isStruct() || declaration->isClass()) {
      userTypePrinter_.print(declaration);

      for (const auto &decl : declaration->decls()) {
        if (auto field = llvm::dyn_cast<clang::FieldDecl>(decl)) {
          memberInfoPrinter_.print(field, "field");
        } else if (auto var = llvm::dyn_cast<clang::VarDecl>(decl)) {
          if (var->isStaticDataMember()) {
            memberInfoPrinter_.print(var, "static");
          }
        } else if (auto method = llvm::dyn_cast<clang::CXXMethodDecl>(decl)) {
          memberInfoPrinter_.print(method, "method");
        }
      }
      llvm::outs() << '\n';
    }
    return true;
  }

private:
  clang::ASTContext *context_;
  MemberInfoPrinter memberInfoPrinter_;
  UserTypePrinter userTypePrinter_;
};

class ClassMembersConsumer final : public clang::ASTConsumer {
public:
  explicit ClassMembersConsumer(clang::ASTContext *context) : visitor_(context) {}

  void HandleTranslationUnit(clang::ASTContext &context) override {
    visitor_.TraverseDecl(context.getTranslationUnitDecl());
  }

private:
  ClassMembersPrinter visitor_;
};

class ClassFieldPrinterAction final : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &ci, llvm::StringRef) override {
    return std::make_unique<ClassMembersConsumer>(&ci.getASTContext());
  }

  bool ParseArgs(const clang::CompilerInstance &ci, const std::vector<std::string> &args) override {
    return true;
  }
};

} // namespace

static clang::FrontendPluginRegistry::Add<ClassFieldPrinterAction>
    X("class-field-printer", "Prints all members of the class");
