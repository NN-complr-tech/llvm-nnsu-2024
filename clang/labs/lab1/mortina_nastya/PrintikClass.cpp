#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

class PrintClassFieldsVisitor : public clang::RecursiveASTVisitor<PrintClassFieldsVisitor> {
public:
    bool VisitCXXRecordDecl(clang::CXXRecordDecl *declaration) {
        if (declaration->isClass() || declaration->isStruct()) {
            llvm::outs() << "Class Name: " << declaration->getNameAsString() << "\n";
            for (auto field : declaration->fields()) {
                llvm::outs() << "|_" << field->getNameAsString() << "\n";
            }
            llvm::outs() << "\n";
        }
        return true;
    }
};

class PrintClassFieldsConsumer : public clang::ASTConsumer {
public:
    PrintClassFieldsVisitor Visitor;
  
    void HandleTranslationUnit(clang::ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }
};

class PrintClassFieldsAction : public clang::PluginASTAction {
public:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI,
                                                 llvm::StringRef) override {
        return std::make_unique<PrintClassFieldsConsumer>();
    }

    bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &Args) override {
        return true;
    }
};

static clang::FrontendPluginRegistry::Add<PrintClassFieldsAction>
    X("prin-elds", "Prints names of all classes and their fields");
    

 