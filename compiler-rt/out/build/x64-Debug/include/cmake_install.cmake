# Install script for directory: D:/llvm/llvm-nnsu-2024/compiler-rt/include

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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xcompiler-rt-headersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sanitizer" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ FILES
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/allocator_interface.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/asan_interface.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/common_interface_defs.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/coverage_interface.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/dfsan_interface.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/hwasan_interface.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/linux_syscall_hooks.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/lsan_interface.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/msan_interface.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/netbsd_syscall_hooks.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/scudo_interface.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/tsan_interface.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/tsan_interface_atomic.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/ubsan_interface.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xcompiler-rt-headersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/fuzzer" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ FILES "D:/llvm/llvm-nnsu-2024/compiler-rt/include/fuzzer/FuzzedDataProvider.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xcompiler-rt-headersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sanitizer" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ FILES "D:/llvm/llvm-nnsu-2024/compiler-rt/include/sanitizer/memprof_interface.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xcompiler-rt-headersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/xray" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ FILES
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/xray/xray_interface.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/xray/xray_log_interface.h"
    "D:/llvm/llvm-nnsu-2024/compiler-rt/include/xray/xray_records.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xcompiler-rt-headersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/orc" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ FILES "D:/llvm/llvm-nnsu-2024/compiler-rt/include/orc_rt/c_api.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xcompiler-rt-headersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/profile" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ FILES "D:/llvm/llvm-nnsu-2024/compiler-rt/include/profile/InstrProfData.inc")
endif()

