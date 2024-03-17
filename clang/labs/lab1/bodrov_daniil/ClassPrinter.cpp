#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

class MemberInfoPrinter {
public:
  void print(const ValueDecl *member, const std::string &memberType) {
    llvm::outs() << "|_ " << member->getNameAsString() << ' ';
    llvm::outs() << '(' << member->getType().getAsString() << '|';
    llvm::outs() << getAccessSpecifierAsString(member) << (memberType == "field" ? ")" : ("|" + memberType + ")") ) << "\n";
  }

private:
  std::string getAccessSpecifierAsString(const ValueDecl *member) {
    switch (member->getAccess()) {
    case AS_public:
      return "public";
    case AS_protected:
      return "protected";
    case AS_private:
      return "private";
    default:
      return "unknown";
    }
  }
};

class UserTypePrinter {
public:
  void print(CXXRecordDecl *userType) {
    llvm::outs() << userType->getNameAsString() << ' ';
    llvm::outs() << (userType->isStruct() ? "(struct" : "(class");
    llvm::outs() << (userType->isTemplated() ? "|template)" : ")") << '\n';
  }
};

class ClassMembersPrinter final : public RecursiveASTVisitor<ClassMembersPrinter> {
public:
  explicit ClassMembersPrinter(ASTContext *context) : context_(context) {}

  bool VisitCXXRecordDecl(CXXRecordDecl *declaration) {
    if (declaration->isStruct() || declaration->isClass()) {
      userTypePrinter_.print(declaration);

      for (const auto &decl : declaration->decls()) {
        if (auto field = llvm::dyn_cast<FieldDecl>(decl)) {
          memberInfoPrinter_.print(field, "field");
        } else if (auto var = llvm::dyn_cast<VarDecl>(decl)) {
          if (var->isStaticDataMember()) {
            memberInfoPrinter_.print(var, "static");
          }
        } else if (auto method = llvm::dyn_cast<CXXMethodDecl>(decl)) {
          memberInfoPrinter_.print(method, "method");
        }
      }
      llvm::outs() << '\n';
    }
    return true;
  }

private:
  ASTContext *context_;
  MemberInfoPrinter memberInfoPrinter_;
  UserTypePrinter userTypePrinter_;
};

class ClassMembersConsumer final : public ASTConsumer {
public:
  explicit ClassMembersConsumer(ASTContext *context) : visitor_(context) {}

  void HandleTranslationUnit(ASTContext &context) override {
    visitor_.TraverseDecl(context.getTranslationUnitDecl());
  }

private:
  ClassMembersPrinter visitor_;
};

class ClassFieldPrinterAction final : public PluginASTAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &ci, llvm::StringRef) override {
    return std::make_unique<ClassMembersConsumer>(&ci.getASTContext());
  }

  bool ParseArgs(const CompilerInstance &ci, const std::vector<std::string> &args) override {
    return true;
  }
};

static FrontendPluginRegistry::Add<ClassFieldPrinterAction>
    X("class-field-printer", "Prints all members of the class");
