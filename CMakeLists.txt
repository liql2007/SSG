cmake_minimum_required(VERSION 2.8)

project(ssg)
include_directories(${PROJECT_SOURCE_DIR}/include)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/module)

# Boost
set(BOOST_ROOT ${PROJECT_SOURCE_DIR}/../_external/usr/local/)
message(${BOOST_ROOT})
find_package(Boost REQUIRED)
if (Boost_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Boost_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Boost_CXX_FLAGS}")
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Boost dynamic-bitset is required")
endif()

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP is required")
endif()

# Python binding
# add_subdirectory(third_party/pybind11)


# Compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
add_definitions(-std=c++11 -O3 -lboost -march=native -Wall -DINFO)

# Source codes
add_subdirectory(src)
add_subdirectory(tests)