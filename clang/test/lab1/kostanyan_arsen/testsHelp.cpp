// RUN: %clang_cc1 -load %llvmshlibdir/PrintNamesPlugin%pluginext -plugin classprinter -plugin-arg-classprinter --help %s 2>&1 | FileCheck %s --check-prefix=HELP
// HELP: This plugin traverses the Abstract Syntax Tree (AST) of a codebase and prints the name and fields of each class it encounters
// HELP-NOT: |_