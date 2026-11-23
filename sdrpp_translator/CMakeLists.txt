# SPDX-License-Identifier: AGPL-3.0-or-later
cmake_minimum_required(VERSION 3.13)
project(translator)

file(GLOB_RECURSE SRC "src/*.cpp")

# In-tree: SDRPP_MODULE_CMAKE is set by the SDR++ parent build (recommended;
# see scripts/build_module.sh). Out-of-tree: pass
# -DSDRPP_MODULE_CMAKE=/path/to/sdrpp_module.cmake from an installed SDR++
# header tree. Modules are ABI-coupled to the exact SDR++ core they link.
if(NOT DEFINED SDRPP_MODULE_CMAKE)
    message(FATAL_ERROR "SDRPP_MODULE_CMAKE not set: build in-tree, or point at an installed sdrpp_module.cmake")
endif()
include(${SDRPP_MODULE_CMAKE})
target_include_directories(translator PRIVATE "src/")

find_package(PkgConfig REQUIRED)
pkg_check_modules(LWS REQUIRED libwebsockets)
target_include_directories(translator PRIVATE ${LWS_INCLUDE_DIRS})
target_link_libraries(translator PRIVATE ${LWS_LIBRARIES})
