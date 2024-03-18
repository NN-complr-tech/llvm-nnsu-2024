#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Rewrite/Core/Rewriter.h"

class RenameVisitor : public clang::RecursiveASTVisitor<RenameVisitor> {

private:
  clang::Rewriter rewriter;
  std::string cur_name;
  std::string new_name;

public:
  explicit RenameVisitor(clang::Rewriter rewriter, clang::StringRef cur_name,
                         clang::StringRef new_name)
      : rewriter(rewriter), cur_name(cur_name), new_name(new_name) {}

  bool VisitFunctionDecl(clang::FunctionDecl *F) {
    if (F->getName() == cur_name) {
      rewriter.ReplaceText(F->getNameInfo().getSourceRange(), new_name);
    }
    return true;
  }

  bool VisitCallExpr(clang::CallExpr *CE) {
    clang::FunctionDecl *callee = CE->getDirectCallee();
    if (callee && callee->getName() == cur_name) {
      rewriter.ReplaceText(CE->getCallee()->getSourceRange(), new_name);
    }
    return true;
  }

  bool VisitVarDecl(clang::VarDecl *VD) {
    if (VD->getName() == cur_name) {
      rewriter.ReplaceText(VD->getLocation(), cur_name.size(), new_name);
    }
    if (VD->getType().getAsString() == cur_name + " *") {
      rewriter.ReplaceText(VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc(),
                           cur_name.size(), new_name);
    }
    if (VD->getType().getAsString() == cur_name) {
      rewriter.ReplaceText(
          VD->getTypeSourceInfo()->getTypeLoc().getSourceRange(), new_name);
    }
    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr *DRE) {
    clang::VarDecl *VD = clang::dyn_cast<clang::VarDecl>(DRE->getDecl());
    if (VD && VD->getName() == cur_name) {
      rewriter.ReplaceText(DRE->getSourceRange(), new_name);
    }
    return true;
  }

  bool VisitCXXRecordDecl(clang::CXXRecordDecl *CXXRD) {
    if (CXXRD->getName() == cur_name) {
      rewriter.ReplaceText(CXXRD->getLocation(), new_name);
      const auto *destructor = CXXRD->getDestructor();
      if (destructor) {
        rewriter.ReplaceText(destructor->getLocation(), cur_name.size() + 1,
                             '~' + new_name);
      }
    }
    return true;
  }

  bool VisitCXXConstructorDecl(clang::CXXConstructorDecl *CD) {
    if (CD->getNameAsString() == cur_name) {
      rewriter.ReplaceText(CD->getLocation(), cur_name.size(), new_name);
    }
    return true;
  }

  bool VisitCXXNewExpr(clang::CXXNewExpr *NE) {
    if (NE->getConstructExpr()->getType().getAsString() == cur_name) {
      rewriter.ReplaceText(NE->getExprLoc(), cur_name.size() + 4,
                           "new " + new_name);
    }
    return true;
  }

  bool save_changes() { return rewriter.overwriteChangedFiles(); }
};

class RenameASTConsumer : public clang::ASTConsumer {

private:
  RenameVisitor Visitor;

public:
  explicit RenameASTConsumer(clang::CompilerInstance &CI,
                             clang::StringRef cur_name,
                             clang::StringRef new_name)
      : Visitor(clang::Rewriter(CI.getSourceManager(), CI.getLangOpts()),
                cur_name, new_name) {}

  void HandleTranslationUnit(clang::ASTContext &context) override {
    Visitor.TraverseDecl(context.getTranslationUnitDecl());
    if (Visitor.save_changes()) {
      llvm::errs() << "An error occurred while saving changes to a file!\n";
    }
  }
};

class VeselRenamePlugin : public clang::PluginASTAction {

private:
  std::string cur_name;
  std::string new_name;

protected:
  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &args) override {

    cur_name = args[0];
    new_name = args[1];

    if (cur_name.find("=") == 0 || cur_name.find("=") == std::string::npos) {
      llvm::errs()
          << "Error -plugin-arg-rename cur-name=\"Current identifier name\""
          << "\n";
    }
    if (new_name.find("=") == 0 || new_name.find("=") == std::string::npos) {
      llvm::errs()
          << "Error -plugin-arg-rename new-name=\"New identifier name\""
          << "\n";
    }

    cur_name = cur_name.substr(cur_name.find("=") + 1);
    new_name = new_name.substr(new_name.find("=") + 1);

    return true;
  }

public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    clang::StringRef InFile) override {
    return std::make_unique<RenameASTConsumer>(CI, cur_name, new_name);
  }
};

static clang::FrontendPluginRegistry::Add<VeselRenamePlugin>
    X("rename", "Rename variable, function or class");
