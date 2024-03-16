#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

namespace {

    class MyASTConsumer : public ASTConsumer {
        CompilerInstance& Instance;

    public:
        MyASTConsumer(CompilerInstance& Instance)
            : Instance(Instance) {}

        void HandleTranslationUnit(ASTContext& context) override {
            struct Visitor : public RecursiveASTVisitor<Visitor> {

                bool VisitCXXRecordDecl(CXXRecordDecl* CxxRDecl) {
                    if (CxxRDecl->isClass() || CxxRDecl->isStruct()) {
                        llvm::outs() << CxxRDecl->getNameAsString() << "\n";

                        for (auto it = CxxRDecl->decls_begin(); it != CxxRDecl->decls_end(); ++it)
                            if (FieldDecl* nDecl = dyn_cast<FieldDecl>(*it))
                                llvm::outs() << "|_ " << nDecl->getNameAsString() << "\n";

                        llvm::outs() << "\n";

                    }
                    return true;
                }
            } v;
            v.TraverseDecl(context.getTranslationUnitDecl());
        }
    };

    class PrintFunctionNamesAction : public PluginASTAction {
    protected:
        std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& CI,
            llvm::StringRef) override {
            return std::make_unique<MyASTConsumer>(CI);
        }

        bool ParseArgs(const CompilerInstance& CI,
            const std::vector<std::string>& args) override {
            return true;
        }
    };

}

static FrontendPluginRegistry::Add<PrintFunctionNamesAction>
X("classprinter", "prints names of classes and their fields");