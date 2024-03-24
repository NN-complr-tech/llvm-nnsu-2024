# Install script for directory: D:/llvm/llvm-nnsu-2024/compiler-rt/lib/asan

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "D:/llvm/llvm-nnsu-2024/compiler-rt/out/install/x64-Debug")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xclang_rt.asan-x86_64x" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/windows" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/compiler-rt/out/build/x64-Debug/lib/windows/clang_rt.asan-x86_64.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xclang_rt.asan_cxx-x86_64x" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/windows" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/compiler-rt/out/build/x64-Debug/lib/windows/clang_rt.asan_cxx-x86_64.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xclang_rt.asan_static-x86_64x" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/windows" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/compiler-rt/out/build/x64-Debug/lib/windows/clang_rt.asan_static-x86_64.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xclang_rt.asan-preinit-x86_64x" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/windows" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/compiler-rt/out/build/x64-Debug/lib/windows/clang_rt.asan-preinit-x86_64.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xclang_rt.asan-dynamic-x86_64x" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/windows" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/llvm/llvm-nnsu-2024/compiler-rt/out/build/x64-Debug/lib/windows/clang_rt.asan_dynamic-x86_64.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xclang_rt.asan-dynamic-x86_64x" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/windows" TYPE SHARED_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/compiler-rt/out/build/x64-Debug/lib/windows/clang_rt.asan_dynamic-x86_64.dll")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xclang_rt.asan_dll_thunk-x86_64x" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/windows" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/compiler-rt/out/build/x64-Debug/lib/windows/clang_rt.asan_dll_thunk-x86_64.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xclang_rt.asan_dynamic_runtime_thunk-x86_64x" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/windows" TYPE STATIC_LIBRARY FILES "D:/llvm/llvm-nnsu-2024/compiler-rt/out/build/x64-Debug/lib/windows/clang_rt.asan_dynamic_runtime_thunk-x86_64.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xasanx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share" TYPE FILE FILES "D:/llvm/llvm-nnsu-2024/compiler-rt/lib/asan/asan_ignorelist.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/llvm/llvm-nnsu-2024/compiler-rt/out/build/x64-Debug/lib/asan/scripts/cmake_install.cmake")
endif()

