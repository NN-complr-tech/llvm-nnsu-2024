#include "clang/AST/Decl.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/ADT/StringRef.h"


namespace {

  class AlwaysInlineConsumer : public clang::ASTConsumer {
  public:
    bool HandleTopLevelDecl(clang::DeclGroupRef decl_group) override {
      for (clang::Decl* decl : decl_group) {
        if (clang::isa<clang::FunctionDecl>(decl)) {
          if (decl->getAttr<clang::AlwaysInlineAttr>()) {
            continue;
          }
          clang::Stmt* body = decl->getBody();
          if (body != nullptr) {
            bool cond_found = false;
            for (clang::Stmt* st : body->children()) {
              if (clang::isa<clang::IfStmt>(st)
              || clang::isa<clang::WhileStmt>(st)
              || clang::isa<clang::ForStmt>(st)
              || clang::isa<clang::DoStmt>(st)
              || clang::isa<clang::SwitchStmt>(st)) {
                cond_found = true;
                break;
              }
            }
            if (!cond_found) {
              // TODO: how to put correct location??
              clang::SourceLocation location(decl->getSourceRange().getBegin());
              clang::SourceRange range(location);
              decl->addAttr(
              clang::AlwaysInlineAttr::Create(decl->getASTContext(), range));
            }
          }
        }
      }
      return true;
    }
  };

  class AlwaysInlinePlugin : public clang::PluginASTAction {
  protected:
    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(
    clang::CompilerInstance& compiler, llvm::StringRef in_file
    ) override {
      return std::make_unique<AlwaysInlineConsumer>();
    }
    bool ParseArgs(
    const clang::CompilerInstance& compiler, const std::vector<std::string>& args
    ) override {
      return true;
    }
  };

} // namespace

static clang::FrontendPluginRegistry::Add<AlwaysInlinePlugin>
X("always-inline", "adds always_inline if no conditions inside body");
