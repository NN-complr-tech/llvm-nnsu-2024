#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Rewrite/Core/Rewriter.h"

class IdentifierRenamer : public clang::RecursiveASTVisitor<IdentifierRenamer> {
    
private:
  clang::Rewriter Rewriter;
  std::string oldName;
  std::string newName;

public:
  explicit IdentifierRenamer(clang::Rewriter rewriter, clang::StringRef oldName,
                             clang::StringRef newName)
      : Rewriter(rewriter), oldName(oldName), newName(newName) {}

  bool VisitFunctionDecl(clang::FunctionDecl *FD) {
    if (FD->getNameAsString() == oldName) {
      Rewriter.ReplaceText(FD->getNameInfo().getSourceRange(), newName);
    }
    return true;
  }

  bool VisitCallExpr(clang::CallExpr *CE) {
    clang::FunctionDecl *callee = CE->getDirectCallee();
    if (callee && callee->getName() == oldName) {
      Rewriter.ReplaceText(CE->getCallee()->getSourceRange(), newName);
    }
    return true;
  }

  bool VisitVarDecl(clang::VarDecl *VD) {
    if (VD->getNameAsString() == oldName) {
      Rewriter.ReplaceText(VD->getLocation(), oldName.size(), newName);
    }
    if (VD->getType().getAsString() == oldName + " *") {
      Rewriter.ReplaceText(VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc(),
                               oldName.size(), newName);
    }
    if (VD->getType().getAsString() == oldName) {
      Rewriter.ReplaceText(
          VD->getTypeSourceInfo()->getTypeLoc().getSourceRange(), newName);
    }
    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr *DRE) {
    clang::VarDecl *VD = clang::dyn_cast<clang::VarDecl>(DRE->getDecl());
    if (VD->getName() == oldName) {
      Rewriter.ReplaceText(DRE->getSourceRange(), newName);
    }
    return true;
  }

  bool VisitCXXRecordDecl(clang::CXXRecordDecl *CXXRD) {
    if (CXXRD->getNameAsString() == oldName) {
      Rewriter.ReplaceText(CXXRD->getLocation(), newName);
      const auto *destructor = CXXRD->getDestructor();
      if (destructor) {
        Rewriter.ReplaceText(destructor->getLocation(), oldName.size() + 1,
                                 '~' + newName);
      }
    }
    return true;
  }

  bool VisitCXXConstructorDecl(clang::CXXConstructorDecl *constructorDecl) {
    if (constructorDecl->getNameAsString() == oldName) {
      Rewriter.ReplaceText(constructorDecl->getLocation(), oldName.size(),
                            newName);
    }
    return true;
  }

  bool VisitCXXNewExpr(clang::CXXNewExpr *newExpr) {
    if (newExpr->getConstructExpr()->getType().getAsString() == oldName) {
      Rewriter.ReplaceText(newExpr->getExprLoc(), oldName.size() + 4,
                            "new " + newName);
    }
    return true;
  }

  bool saveChanges() { return Rewriter.overwriteChangedFiles(); }
};

class RenameASTConsumer : public clang::ASTConsumer {
private:
  IdentifierRenamer visitor;
  
public:
  explicit RenameASTConsumer(clang::CompilerInstance &CI,
                             clang::StringRef oldName,
                             clang::StringRef newName)
      : visitor(clang::Rewriter(CI.getSourceManager(), CI.getLangOpts()),
                oldName, newName) {}

  void HandleTranslationUnit(clang::ASTContext &context) override {
    visitor.TraverseDecl(context.getTranslationUnitDecl());
    if (visitor.saveChanges()) {
      llvm::errs() << "An error occurred while saving changes to a file!\n";
    }
  }
};

class RenamePlugin : public clang::PluginASTAction {
private:
  std::string oldName;
  std::string newName;

protected:
  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    
    std::vector<std::pair<std::string, std::string>> parameters = {
        {"old-name=", ""}, {"new-name=", ""}};

    if (!args.empty() && args[0] == "help") {
      printHelp(llvm::errs());
      return true;
    }

    if (args.size() < 2) {
      printParamsError(CI);
      return false;
    }

    for (const auto &arg : args) {
      bool found = false;
      for (auto &param : parameters) {
        if (arg.find(param.first) == 0 && param.second.empty()) {
          param.second = arg.substr(param.first.size());
          found = true;
          break;
        }
      }
      if (!found) {
        printParamsError(CI);
        return false;
      }
    }

    oldName = parameters[1].second;
    newName = parameters[2].second;
    return true;
  }

  void printHelp(llvm::raw_ostream &os) {
    os << "Specify two required arguments:\n"
          "-plugin-arg-rename old-name=\"Old identifier name\"\n"
          "-plugin-arg-rename new-name=\"New identifier name\"\n";
  }

  void printParamsError(const clang::CompilerInstance &CI) {
    clang::DiagnosticsEngine &diagnosticsEngine = CI.getDiagnostics();

    diagnosticsEngine.Report(
        diagnosticsEngine.getCustomDiagID(clang::DiagnosticsEngine::Error,
                          "Invalid arguments\n"
                          "Specify \"-plugin-arg-rename help\" for usage\n"));
  }

public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    clang::StringRef inFile) override {
    return std::make_unique<RenameASTConsumer>(CI, oldName, newName);
  }
};

static clang::FrontendPluginRegistry::Add<RenamePlugin>
    X("rename", "Rename variable, function or class identifier");
