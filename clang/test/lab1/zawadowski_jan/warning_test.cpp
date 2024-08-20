// RUN: split-file %s %t
// RUN: %clang_cc1 -load %llvmshlibdir/LegacyCodePlugin%pluginext -plugin legacy-code %t/WithNotCheckClass.cpp -plugin-arg-legacy-code -ignoreClassMembers 2>&1 | FileCheck %t/WithNotCheckClass.cpp
// RUN: %clang_cc1 -load %llvmshlibdir/LegacyCodePlugin%pluginext -plugin legacy-code %t/WithoutNotCheckClass.cpp 2>&1 | FileCheck %t/WithoutNotCheckClass.cpp

//--- WithNotCheckClass.cpp

// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
void some_legacy_function();

// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
void SomeLegacyFunction();

// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
void some_deprecated_function();

// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
void SomeDeprecatedFunction();

// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
void some_obsolete_function();

// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
void SomeObsoleteFunction();

// CHECK-NOT: warning: Found potential legacy code usage: function or method contains special word in its name.
void function();

// CHECK-NOT: warning: Found potential legacy code usage: function or method contains special word in its name.
void function_depr();

class CheckClass {
	// CHECK-NOT: warning: Found potential legacy code usage: function or method contains special word in its name.
	void legacy();
	// CHECK-NOT: warning: Found potential legacy code usage: function or method contains special word in its name.
	void deprecated();
	// CHECK-NOT: warning: Found potential legacy code usage: function or method contains special word in its name.
	void obsolete();
	// CHECK-NOT: warning: Found potential legacy code usage: function or method contains special word in its name.
	void function();
};

//--- WithoutNotCheckClass.cpp

// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
void some_legacy_function();

// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
void SomeLegacyFunction();

// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
void some_deprecated_function();

// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
void SomeDeprecatedFunction();

// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
void some_obsolete_function();

// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
void SomeObsoleteFunction();

// CHECK-NOT: warning: Found potential legacy code usage: function or method contains special word in its name.
void function();

// CHECK-NOT: warning: Found potential legacy code usage: function or method contains special word in its name.
void function_depr();

class CheckClass {
	// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
	void legacy();
	// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
	void deprecated();
	// CHECK: warning: Found potential legacy code usage: function or method contains special word in its name.
	void obsolete();
	// CHECK-NOT: warning: Found potential legacy code usage: function or method contains special word in its name.
	void function();
};