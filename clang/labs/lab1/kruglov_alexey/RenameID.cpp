#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/Sequence.h"

class RenameVisitor : public clang::RecursiveASTVisitor<RenameVisitor> {
private:
  clang::Rewriter Rewriter;
  std::string OldName;
  std::string NewName;

public:
  explicit RenameVisitor(clang::Rewriter Rewriter, std::string OldName,
                         std::string NewName)
      : Rewriter(Rewriter), OldName(OldName), NewName(NewName){};

  bool VisitFunctionDecl(clang::FunctionDecl *Func) {
    std::string Name = Func->getNameAsString();

    if (Name == OldName) {
      Rewriter.ReplaceText(Func->getNameInfo().getSourceRange(), NewName);
      Rewriter.overwriteChangedFiles();
    }
    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr *Expr) {
    std::string Name = Expr->getNameInfo().getAsString();

    if (Name == OldName) {
      Rewriter.ReplaceText(Expr->getNameInfo().getSourceRange(), NewName);
      Rewriter.overwriteChangedFiles();
    }
    return true;
  }

  bool VisitVarDecl(clang::VarDecl *Var) {
    std::string Name = Var->getNameAsString();

    if (Name == OldName) {
      Rewriter.ReplaceText(Var->getLocation(), Name.size(), NewName);
      Rewriter.overwriteChangedFiles();
    }

    if (Var->getType().getAsString() == OldName || 
        Var->getType().getAsString() == OldName + " *" ||
        Var->getType().getAsString() == OldName + " &") {
      Rewriter.ReplaceText(Var->getTypeSourceInfo()->getTypeLoc().getBeginLoc(),
                           OldName.size(), NewName);
      Rewriter.overwriteChangedFiles();
    }

    return true;
  }

  bool VisitCXXRecordDecl(clang::CXXRecordDecl *RecordDecl) {
    std::string Name = RecordDecl->getNameAsString();

    if (Name == OldName) {
      Rewriter.ReplaceText(RecordDecl->getLocation(), Name.size(), NewName);

      const clang::CXXDestructorDecl *RecordDestr = RecordDecl->getDestructor();
      if (RecordDestr)
        Rewriter.ReplaceText(RecordDestr->getLocation(), Name.size() + 1,
                             "~" + NewName);

      Rewriter.overwriteChangedFiles();
    }
    return true;
  }

  bool VisitCXXNewExpr(clang::CXXNewExpr *NewExpr) {
    std::string Name = NewExpr->getConstructExpr()->getType().getAsString();

    if (Name == OldName) {
      Rewriter.ReplaceText(NewExpr->getExprLoc(), Name.size() + 4,
                           "new " + NewName);
      Rewriter.overwriteChangedFiles();
    }
    return true;
  }
};

class RenameIDConsumer : public clang::ASTConsumer {
protected:
  RenameVisitor visitor;

public:
  explicit RenameIDConsumer(clang::CompilerInstance &CI, 
                            std::string OldName,
                            std::string NewName)
      : visitor(clang::Rewriter(CI.getSourceManager(), CI.getLangOpts()),
                OldName, NewName) {}

  void HandleTranslationUnit(clang::ASTContext &Context) override {
    visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class RenameIDPlugin : public clang::PluginASTAction {
private:
  std::string OldName;
  std::string NewName;

protected:
  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    if (args[0].find("OldName=") == std::string::npos || args[1].find("NewName=") == std::string::npos){
      llvm::errs() << "Error in parameters input.\n"
                      "Format of input:\n"
                      "OldName='MyOldName'\n"
                      "NewName='MyNewName'\n";
      return true;
    }

    OldName = args[0].substr(args[0].find("=") + 1);
    NewName = args[1].substr(args[1].find("=") + 1);

    return true;
  }

public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    clang::StringRef InFile) override {
    return std::make_unique<RenameIDConsumer>(CI, OldName, NewName);
  }
};

static clang::FrontendPluginRegistry::Add<RenameIDPlugin>
    X("RenameID", "Rename variable, function or class identificators");
