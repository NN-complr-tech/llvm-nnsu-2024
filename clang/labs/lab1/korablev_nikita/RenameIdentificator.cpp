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
    StringRef oldName;
    StringRef newName;
public:
    explicit RenameVisitor(Rewriter Rewriter, StringRef OldName,
                                 StringRef NewName) : rewriter(Rewriter), oldName(OldName), newName(NewName) {};

    bool VisitFunctionDecl(FunctionDecl *FD) {
        llvm::outs() << FD->getName() << " | ";
        // if (FD->getNameAsString() == oldName) {
        //     rewriter.ReplaceText(FD->getNameInfo().getSourceRange(), newName);
        //     rewriter.overwriteChangedFiles();
        // }


        llvm::outs() << FD->getName() << "\n";
        return true;
    }
};

class RenameIdConsumer : public ASTConsumer {
    RenameVisitor visitor;
public:
    explicit RenameIdConsumer(CompilerInstance &CI, StringRef oldName, StringRef newName): 
                visitor(clang::Rewriter(CI.getSourceManager(), CI.getLangOpts()), oldName, newName) {}

    void HandleTranslationUnit(ASTContext &Context) override {
        visitor.TraverseDecl(Context.getTranslationUnitDecl());

        makeRule(declRefExpr(to(functionDecl(hasName("sum")))),
                changeTo(cat("renamed")),
                cat("'sum' has been renamed 'renamed'"));
    }
};

class RenameIdPlugin: public clang::PluginASTAction {
private:
    std::vector<std::string> s = {"sum", "renamed"};
    std::string OldName = s[0];
    std::string NewName = s[1];
protected:
    bool ParseArgs(const clang::CompilerInstance &Compiler,
        const std::vector<std::string> &args) override { return true; }
public:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer (
            clang::CompilerInstance &Compiler, llvm::StringRef InFile) override {
        return std::make_unique<RenameIdConsumer>(Compiler, OldName, NewName);
    }
};

static FrontendPluginRegistry::Add<RenameIdPlugin>
X("renamed-id", "Idetificator was renamed.");