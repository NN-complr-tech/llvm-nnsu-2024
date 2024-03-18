#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Rewrite/Core/Rewriter.h"


class RenameVisitor : public clang::RecursiveASTVisitor<RenameVisitor> {
public:
  explicit RenameVisitor(clang::Rewriter rewriter,
                         clang::StringRef cur_name, clang::StringRef new_name)
      : rewriter(rewriter), cur_name(cur_name), new_name(new_name) {
  }

  bool VisitFunctionDecl(clang::FunctionDecl *func) {
    if (func->getName() == cur_name) {
      rewriter.ReplaceText(func->getNameInfo().getSourceRange(), new_name);
    }
    return true;
  }

  bool VisitCallExpr(clang::CallExpr *call) {
    clang::FunctionDecl *callee = call->getDirectCallee();
    if (callee && callee->getName() == cur_name) {
      rewriter.ReplaceText(call->getCallee()->getSourceRange(), new_name);
    }
    return true;
  }

  bool VisitVarDecl(clang::VarDecl *var) {
    if (var->getName() == cur_name) {
      rewriter.ReplaceText(var->getLocation(), cur_name.size(), new_name);
    }
    if (var->getType().getAsString() == cur_name + " *") {
      rewriter.ReplaceText(var->getTypeSourceInfo()->getTypeLoc().getBeginLoc(),
                           cur_name.size(), new_name);
    }
    if (var->getType().getAsString() == cur_name) {
      rewriter.ReplaceText(
          var->getTypeSourceInfo()->getTypeLoc().getSourceRange(), new_name);
    }
    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr *expr) {
    clang::VarDecl *var = clang::dyn_cast<clang::VarDecl>(expr->getDecl());
    if (var && var->getName() == cur_name) {
      rewriter.ReplaceText(expr->getSourceRange(), new_name);
    }
    return true;
  }

  bool VisitCXXRecordDecl(clang::CXXRecordDecl *record) {
    if (record->getName() == cur_name) {
      rewriter.ReplaceText(record->getLocation(), new_name);
      const auto *destructor = record->getDestructor();
      if (destructor) {
        rewriter.ReplaceText(destructor->getLocation(), cur_name.size() + 1,
                             '~' + new_name);
      }
    }
    return true;
  }

  bool VisitCXXConstructorDecl(clang::CXXConstructorDecl *constructor) {
    if (constructor->getNameAsString() == cur_name) {
      rewriter.ReplaceText(constructor->getLocation(), cur_name.size(),
                           new_name);
    }
    return true;
  }

  bool VisitCXXNewExpr(clang::CXXNewExpr *newExpr) {
    if (newExpr->getConstructExpr()->getType().getAsString() == cur_name) {
      rewriter.ReplaceText(newExpr->getExprLoc(), cur_name.size() + 4,
                           "new " + new_name);
    }
    return true;
  }

  bool save_changes() { return rewriter.overwriteChangedFiles(); }

private:
  clang::Rewriter rewriter;
  std::string cur_name;
  std::string new_name;
};

class RenameASTConsumer : public clang::ASTConsumer {
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

private:
  RenameVisitor Visitor;
};

class VeselRenamePlugin : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    clang::StringRef InFile) override {
    return std::make_unique<RenameASTConsumer>(CI, cur_name, new_name);
  }

protected:
  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    // std::vector<std::pair<std::string, std::string>> params = {
    //     {"cur-name=", ""}, {"new-name=", ""}};

    // if (!args.empty() && args[0] == "help") {
    //   PrintHelp(llvm::errs());
    //   return true;
    // }

    // if (args.size() < 2) {
    //   PrintParamsError(CI);
    //   return false;
    // }

    // for (const auto &arg : args) {
    //   bool is_found = false;
    //   for (auto &param : params) {
    //     if (arg.find(param.first) == 0 && param.second.empty()) {
    //       param.second = arg.substr(param.first.size());
    //       is_found = true;
    //       break;
    //     }
    //   }
    //   if (!is_found) {
    //     PrintParamsError(CI);
    //     return false;
    //   }
    // }

    // cur_name = params[0].second;
    // new_name = params[1].second;

    cur_name = args[0];
    new_name = args[1];

    if (cur_name.find("=") == 0 || cur_name.find("=") == std::string::npos) {
      llvm::errs() << "Error entering the `cur_name` parameter."
                   << "\n";
    }
    if (new_name.find("=") == 0 || new_name.find("=") == std::string::npos) {
      llvm::errs() << "Error entering the `new_name` parameter."
                   << "\n";
    }

    cur_name = cur_name.substr(cur_name.find("=") + 1);
    new_name = new_name.substr(new_name.find("=") + 1);

    return true;
  }

  void PrintHelp(llvm::raw_ostream &ros) {
    ros << "Specify three required arguments:\n"
           "-plugin-arg-rename cur-name=\"Current identifier name\"\n"
           "-plugin-arg-rename new-name=\"New identifier name\"\n";
  }

  void PrintParamsError(const clang::CompilerInstance &CI) {
    clang::DiagnosticsEngine &D = CI.getDiagnostics();

    D.Report(
        D.getCustomDiagID(clang::DiagnosticsEngine::Error,
                          "Invalid arguments\n"
                          "Specify \"-plugin-arg-rename help\" for usage\n"));
  }

private:
  std::string cur_name;
  std::string new_name;
};

static clang::FrontendPluginRegistry::Add<VeselRenamePlugin>
    X("rename", "Rename variable, function or class");
