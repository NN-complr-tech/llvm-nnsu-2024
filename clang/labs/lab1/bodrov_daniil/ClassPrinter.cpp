#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

class MemberInfoPrinter {
public:
  void print(const clang::ValueDecl *Member, const std::string &MemberType) {
    llvm::outs() << "|_ " << Member->getNameAsString() << ' ';
    llvm::outs() << '(' << Member->getType().getAsString() << '|';
    llvm::outs() << getAccessSpecifierAsString(Member)
                 << (MemberType == "field" ? ")" : ("|" + MemberType + ")")) << "\n";
  }

private:
  std::string getAccessSpecifierAsString(const clang::ValueDecl *Member) {
    switch (Member->getAccess()) {
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
  void print(clang::CXXRecordDecl *UserType) {
    llvm::outs() << UserType->getNameAsString() << ' ';
    llvm::outs() << (UserType->isStruct() ? "(struct" : "(class");
    llvm::outs() << (UserType->isTemplated() ? "|template)" : ")") << '\n';
  }
};

class ClassMembersPrinter final
    : public clang::RecursiveASTVisitor<ClassMembersPrinter> {
public:
  explicit ClassMembersPrinter(clang::ASTContext *Context) : Context_(Context) {}

  bool VisitCXXRecordDecl(clang::CXXRecordDecl *Declaration) {
    if (Declaration->isStruct() || Declaration->isClass()) {
      UserTypePrinter_.print(Declaration);

      for (const auto &Decl : Declaration->decls()) {
        if (auto Field = llvm::dyn_cast<clang::FieldDecl>(Decl)) {
          MemberInfoPrinter_.print(Field, "field");
        } else if (auto Var = llvm::dyn_cast<clang::VarDecl>(Decl)) {
          if (Var->isStaticDataMember()) {
            MemberInfoPrinter_.print(Var, "static");
          }
        } else if (auto Method = llvm::dyn_cast<clang::CXXMethodDecl>(Decl)) {
          MemberInfoPrinter_.print(Method, "method");
        }
      }
      llvm::outs() << '\n';
    }
    return true;
  }

private:
  clang::ASTContext *Context_;
  MemberInfoPrinter MemberInfoPrinter_;
  UserTypePrinter UserTypePrinter_;
};

class ClassMembersConsumer final : public clang::ASTConsumer {
public:
  explicit ClassMembersConsumer(clang::ASTContext *Context)
      : Visitor_(Context) {}

  void HandleTranslationUnit(clang::ASTContext &Context) override {
    Visitor_.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  ClassMembersPrinter Visitor_;
};

class ClassFieldPrinterAction final : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef) override {
    return std::make_unique<ClassMembersConsumer>(&CI.getASTContext());
  }

  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &Args) override {
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<ClassFieldPrinterAction>
    X("class-field-printer", "Prints all members of the class");