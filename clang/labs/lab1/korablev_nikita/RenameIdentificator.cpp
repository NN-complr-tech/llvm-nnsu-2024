#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Transformer.h"
#include "clang/Tooling/Transformer/Stencil.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::transformer;

using ::clang::transformer::changeTo;
using ::clang::transformer::makeRule;
using ::clang::transformer::RewriteRuleWith;



class RenameVisitor: public RecursiveASTVisitor<RenameVisitor> {
private:
    Rewriter rewriter;
    std::string oldName;
    std::string newName;
public:
    explicit RenameVisitor(Rewriter Rewriter, std::string OldName,
                                 std::string NewName) : rewriter(Rewriter), oldName(OldName), newName(NewName) {};

    bool VisitFunctionDecl(FunctionDecl *FD) {
        std::string name = FD->getNameAsString();
        
        if (name == oldName) {
            rewriter.ReplaceText(FD->getNameInfo().getSourceRange(), newName);
            rewriter.overwriteChangedFiles();
        }

        return true;
    }

    bool VisitDeclRefExpr(DeclRefExpr *DRE) {
        std::string name = DRE->getNameInfo().getAsString();

        if (name == oldName) {
            rewriter.ReplaceText(DRE->getNameInfo().getSourceRange(), newName);
            rewriter.overwriteChangedFiles();
        }

        return true;
    }

    bool VisitVarDecl(VarDecl *VD) {
        std::string name = VD->getNameAsString();

        if (name == oldName) {
            rewriter.ReplaceText(VD->getLocation(), name.size(), newName);
            rewriter.overwriteChangedFiles();
        }

        if (VD->getType().getAsString() == oldName + " *") {
            rewriter.ReplaceText(VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc(), name.size(), newName);
            rewriter.overwriteChangedFiles();
        }

        if (VD->getType().getAsString() == oldName) {
            rewriter.ReplaceText(VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc(), name.size(), newName);
            rewriter.overwriteChangedFiles();
        }

        return true;
    }

    bool VisitCXXRecordDecl(CXXRecordDecl *CXXRD) {
        std::string name = CXXRD->getNameAsString();

        if (name == oldName) {
            rewriter.ReplaceText(CXXRD->getLocation(), name.size(), newName);

            const CXXDestructorDecl *CXXDD = CXXRD->getDestructor();
            if (CXXDD) rewriter.ReplaceText(CXXDD->getLocation(), name.size() + 1, "~" + newName);

            rewriter.overwriteChangedFiles();
        }

        return true;
    }

    bool VisitCXXNewExpr(CXXNewExpr *CXXNE) {
        std::string name = CXXNE->getConstructExpr()->getType().getAsString();

        if (name == oldName) {
            rewriter.ReplaceText(CXXNE->getExprLoc(), name.size() + 4, "new " + newName);
            rewriter.overwriteChangedFiles();
        }

        return true;
    }
};

class RenameIdConsumer : public ASTConsumer {
    RenameVisitor visitor;
public:
    explicit RenameIdConsumer(CompilerInstance &CI, std::string oldName, std::string newName): 
                visitor(clang::Rewriter(CI.getSourceManager(), CI.getLangOpts()), oldName, newName) {}

    void HandleTranslationUnit(ASTContext &Context) override {
        visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }
};

class RenameIdPlugin: public clang::PluginASTAction {
private:
    std::string OldName;
    std::string NewName;
protected:
    bool ParseArgs(const clang::CompilerInstance &Compiler,
        const std::vector<std::string> &args) override {
            OldName = args[0];
            NewName = args[1];

            if (OldName.find("=") == 0 || OldName.find("=") == std::string::npos) {
                llvm::errs() << "Error entering the `OldName` parameter." << "\n";
            }
            if (NewName.find("=") == 0 || NewName.find("=") == std::string::npos) {
                llvm::errs() << "Error entering the `NewName` parameter." << "\n";
            }

            OldName = OldName.substr(OldName.find("=") + 1);
            NewName = NewName.substr(NewName.find("=") + 1);

            return true;
        }
public:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer (
            clang::CompilerInstance &Compiler, llvm::StringRef InFile) override {
        return std::make_unique<RenameIdConsumer>(Compiler, OldName, NewName);
    }
};

static FrontendPluginRegistry::Add<RenameIdPlugin>
X("renamed-id", "Idetificator was renamed.");