#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/ADT/StringRef.h"
#include <memory>


namespace {

  class AlwaysInlineConsumer : public clang::ASTConsumer {
  public:
    // OLD, do nothing, can safely ignore or delete
    void HandleTranslationUnitOLD(clang::ASTContext& context) {
      struct Visitor : public clang::RecursiveASTVisitor<Visitor> {
        clang::ASTContext& context;
        Visitor(clang::ASTContext& context) : context(context) {}
        bool VisitFunctionDecl(clang::FunctionDecl* func) {
          if (func->getAttr<clang::AlwaysInlineAttr>() != nullptr) {
            return true;
          }
          clang::Stmt* body = func->getBody();
          bool cond_found = false;
          if (body != nullptr) {
            for (clang::Stmt* st : body->children()) {
              // pls don't use goto
              // no switch?
              if (clang::isa<clang::IfStmt>(st) || clang::isa<clang::WhileStmt>(st) || clang::isa<clang::ForStmt>(st)) {
                cond_found = true;
                break;
              }
            }
            if (!cond_found) {
              if (func->hasAttr<clang::NoInlineAttr>()) {
                func->dropAttr<clang::NoInlineAttr>();
              }
              func->addAttr(clang::AlwaysInlineAttr::CreateImplicit(context));
            }
          }
          return true;
        }
      } visitor(context);
      visitor.TraverseDecl(context.getTranslationUnitDecl());
    }
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
              if (clang::isa<clang::IfStmt>(st) || clang::isa<clang::WhileStmt>(st) || clang::isa<clang::ForStmt>(st)) {
                cond_found = true;
                break;
              }
            }
            if (!cond_found) {
              if (decl->hasAttr<clang::NoInlineAttr>()) {
                //decl->dropAttr<clang::NoInlineAttr>();
              }
              // TODO: wrap in ifs
              decl->dropAttr<clang::OptimizeNoneAttr>();
              decl->dropAttr<clang::NoInlineAttr>();
              // TODO: how to put correct location??
              clang::SourceLocation location(decl->getSourceRange().getBegin());
              clang::SourceRange range(location);
              decl->addAttr(clang::AlwaysInlineAttr::Create(decl->getASTContext(), range));
            }
          }
        }
      }
      return true;
    }
  };

  class AlwaysInlinePlugin : public clang::PluginASTAction {
  protected:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& compiler, llvm::StringRef in_file) override {
      return std::make_unique<AlwaysInlineConsumer>();
    }
    bool ParseArgs(const clang::CompilerInstance& compiler, const std::vector<std::string>& args) override {
      return true;
    }
  };

} // namespace

static clang::FrontendPluginRegistry::Add<AlwaysInlinePlugin> X("always-inline", "adds always_inline if no conditions inside function's body");
