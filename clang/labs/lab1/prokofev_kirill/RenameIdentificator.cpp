#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

class RenameVisitor : public clang::RecursiveASTVisitor<RenameVisitor> {
private:
  clang::Rewriter rewriter;
  std::string oldName;
  std::string newName;

public:
  explicit RenameVisitor(clang::Rewriter rewriter, std::string oldName,
                         std::string newName)
      : rewriter(rewriter), oldName(oldName), newName(newName){};

  bool VisitFunctionDecl(clang::FunctionDecl *FD) {
    if (FD->getNameAsString() == oldName) {
      rewriter.ReplaceText(FD->getNameInfo().getSourceRange(), newName);
      SaveChanges();
    }
    return true;
  }

  bool SaveChanges() { return rewriter.overwriteChangedFiles(); }

  bool VisitDeclRefExpr(clang::DeclRefExpr *DRE) {
    if (DRE->getNameInfo().getAsString() == oldName) {
      rewriter.ReplaceText(DRE->getNameInfo().getSourceRange(), newName);
      SaveChanges();
    }
    return true;
  }

  bool VisitVarDecl(clang::VarDecl *VD) {

    if (VD->getNameAsString() == oldName) {
      rewriter.ReplaceText(VD->getLocation(), VD->getNameAsString().size(),
                           newName);
      rewriter.overwriteChangedFiles();
    }
    if (VD->getType().getAsString() == oldName + " *" ||
        VD->getType().getAsString() == oldName) {
      rewriter.ReplaceText(VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc(),
                           VD->getNameAsString().size(), newName);
      SaveChanges();
    }
    return true;
  }

  bool VisitCXXRecordDecl(clang::CXXRecordDecl *CXXRD) {
    if (CXXRD->getNameAsString() == oldName) {
      rewriter.ReplaceText(CXXRD->getLocation(),
                           CXXRD->getNameAsString().size(), newName);

      const clang::CXXDestructorDecl *CXXDD = CXXRD->getDestructor();
      if (CXXDD)
        rewriter.ReplaceText(CXXDD->getLocation(),
                             CXXRD->getNameAsString().size() + 1,
                             "~" + newName);
      SaveChanges();
    }
    return true;
  }

  bool VisitCXXNewExpr(clang::CXXNewExpr *CXXNE) {
    if (CXXNE->getConstructExpr()->getType().getAsString() == oldName) {
      rewriter.ReplaceText(
          CXXNE->getExprLoc(),
          CXXNE->getConstructExpr()->getType().getAsString().size() + 4,
          "new " + newName);
      SaveChanges();
    }
    return true;
  }
};

class RenameConsumer : public clang::ASTConsumer {
  RenameVisitor visitor;

public:
  explicit RenameConsumer(clang::CompilerInstance &CI, std::string oldName,
                          std::string newName)
      : visitor(clang::Rewriter(CI.getSourceManager(), CI.getLangOpts()),
                oldName, newName) {}

  void HandleTranslationUnit(clang::ASTContext &Context) override {
    visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class RenamePlugin : public clang::PluginASTAction {
private:
  std::string oldName;
  std::string newName;

protected:
  bool ParseArgs(const clang::CompilerInstance &Compiler,
                 const std::vector<std::string> &args) override {
    std::vector<std::pair<std::string, std::string>> params = {
        {"oldname=", ""}, {"newname=", ""}};
    oldName = args[0];
    newName = args[1];

    if (oldName.find("=") == 0 || oldName.find("=") == std::string::npos) {
      llvm::errs() << "Invalid arguments."
                   << "\n";
    }
    if (newName.find("=") == 0 || newName.find("=") == std::string::npos) {
      llvm::errs() << "Invalid arguments."
                   << "\n";
    }
    oldName = oldName.substr(oldName.find("=") + 1);
    newName = newName.substr(newName.find("=") + 1);
    return true;
  }

public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<RenameConsumer>(Compiler, oldName, newName);
  }
};

static clang::FrontendPluginRegistry::Add<RenamePlugin>
    X("renameplug", "Plugin for renaming: variables, functions, classes");