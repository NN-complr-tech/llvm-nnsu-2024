#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Rewrite/Core/Rewriter.h"

enum class IdType { Var, Func, Class };

class RenameVisitor : public clang::RecursiveASTVisitor<RenameVisitor> {
private:
  clang::Rewriter rewriter;
  IdType type;
  std::string currentName;
  std::string newName;

public:
  explicit RenameVisitor(clang::Rewriter rewriter, IdType type,
                         clang::StringRef currentName, clang::StringRef newName)
      : rewriter(rewriter), type(type), currentName(currentName), newName(newName) {
  }

  bool VisitFunctionDecl(clang::FunctionDecl *func) {
    if (type == IdType::Func && func->getName() == currentName) {
      rewriter.ReplaceText(func->getNameInfo().getSourceRange(), newName);
    }
    return true;
  }

  bool VisitCallExpr(clang::CallExpr *call) {
    if (type == IdType::Func) {
      clang::FunctionDecl *callee = call->getDirectCallee();
      if (callee && callee->getName() == currentName) {
        rewriter.ReplaceText(call->getCallee()->getSourceRange(), newName);
      }
    }
    return true;
  }

  bool VisitVarDecl(clang::VarDecl *var) {
    if (type == IdType::Var && var->getName() == currentName) {
      rewriter.ReplaceText(var->getLocation(), currentName.size(), newName);
    }
    if (type == IdType::Class &&
        var->getType().getAsString() == currentName + " *") {
      rewriter.ReplaceText(var->getTypeSourceInfo()->getTypeLoc().getBeginLoc(),
                           currentName.size(), newName);
    }
    if (type == IdType::Class && var->getType().getAsString() == currentName) {
      rewriter.ReplaceText(
          var->getTypeSourceInfo()->getTypeLoc().getSourceRange(), newName);
    }
    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr *expr) {
    clang::VarDecl *var = clang::dyn_cast<clang::VarDecl>(expr->getDecl());
    if (type == IdType::Var && var && var->getName() == currentName) {
      rewriter.ReplaceText(expr->getSourceRange(), newName);
    }
    return true;
  }

  bool VisitCXXRecordDecl(clang::CXXRecordDecl *record) {
    if (type == IdType::Class && record->getName() == currentName) {
      rewriter.ReplaceText(record->getLocation(), newName);
      const auto *destructor = record->getDestructor();
      if (destructor) {
        rewriter.ReplaceText(destructor->getLocation(), currentName.size() + 1,
                             '~' + newName);
      }
    }
    return true;
  }

  bool VisitCXXConstructorDecl(clang::CXXConstructorDecl *constructor) {
    if (type == IdType::Class) {
      if (constructor->getNameAsString() == currentName) {
        rewriter.ReplaceText(constructor->getLocation(), currentName.size(),
                             newName);
      }
    }
    return true;
  }

  bool VisitCXXNewExpr(clang::CXXNewExpr *newExpr) {
    if (type == IdType::Class) {
      if (newExpr->getConstructExpr()->getType().getAsString() == currentName) {
        rewriter.ReplaceText(newExpr->getExprLoc(), currentName.size() + 4,
                             "new " + newName);
      }
    }
    return true;
  }

  bool save_changes() { return rewriter.overwriteChangedFiles(); }


};

class RenameASTConsumer : public clang::ASTConsumer {
private:
  RenameVisitor Visitor;

public:
  explicit RenameASTConsumer(clang::CompilerInstance &CI, IdType type,
                             clang::StringRef currentName,
                             clang::StringRef newName)
      : Visitor(clang::Rewriter(CI.getSourceManager(), CI.getLangOpts()), type,
                currentName, newName) {}

  void HandleTranslationUnit(clang::ASTContext &context) override {
    Visitor.TraverseDecl(context.getTranslationUnitDecl());
    if (Visitor.save_changes()) {
      llvm::errs() << "An error occurred while saving changes to a file!\n";
    }
  }


};

class RenamePlugin : public clang::PluginASTAction {
private:
  IdType type;
  std::string currentName;
  std::string newName;

public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    clang::StringRef InFile) override {
    return std::make_unique<RenameASTConsumer>(CI, type, currentName, newName);
  }

protected:
  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    std::vector<std::pair<std::string, std::string>> params = {
        {"type=", ""}, {"cur-name=", ""}, {"new-name=", ""}};

    if (!args.empty() && args[0] == "help") {
      PrintHelp(llvm::errs());
      return true;
    }

    if (args.size() < 3) {
      PrintParamsError(CI);
      return false;
    }

    for (const auto &arg : args) {
      bool is_found = false;
      for (auto &param : params) {
        if (arg.find(param.first) == 0 && param.second.empty()) {
          param.second = arg.substr(param.first.size());
          is_found = true;
          break;
        }
      }
      if (!is_found) {
        PrintParamsError(CI);
        return false;
      }
    }

    std::vector<std::pair<std::string, IdType>> id_type = {
        {"var", IdType::Var}, {"func", IdType::Func}, {"class", IdType::Class}};
    size_t i;
    for (i = 0; i < id_type.size(); i++) {
      if (params[0].second == id_type[i].first) {
        type = id_type[i].second;
        break;
      }
    }
    if (i == id_type.size()) {
      PrintParamsError(CI);
      return false;
    }
    currentName = params[1].second;
    newName = params[2].second;
    return true;
  }

  void PrintHelp(llvm::raw_ostream &ros) {
    ros << "Specify three required arguments:\n"
           "-plugin-arg-Rename type=[\"var\", \"func\", \"class\"]\n"
           "-plugin-arg-Rename cur-name=\"Current identifier name\"\n"
           "-plugin-arg-Rename new-name=\"New identifier name\"\n";
  }

  void PrintParamsError(const clang::CompilerInstance &CI) {
    clang::DiagnosticsEngine &D = CI.getDiagnostics();

    D.Report(
        D.getCustomDiagID(clang::DiagnosticsEngine::Error,
                          "Invalid arguments\n"
                          "Specify \"-plugin-arg-Rename help\" for usage\n"));
  }


};

static clang::FrontendPluginRegistry::Add<RenamePlugin>
    X("Rename", "Rename variable, function or class");
