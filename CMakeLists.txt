cmake_minimum_required(VERSION 3.13)
project(bitonic_sort)
find_package(OpenCL REQUIRED)

add_subdirectory(bitonic_sort)
add_subdirectory(tests)

option(ENABLE_ASAN OFF)

target_compile_options(bitonic_sort PRIVATE
$<$<CONFIG:Debug>:
    -Wall
    -Wextra
    -Wpedantic
    -DDEBUG
    -Wmissing-declarations
    -Wcast-align
    -Wshadow
    -Wsign-conversion>
)

if(ENABLE_ASAN)
    foreach(target bitonic_sort tests)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
        target_link_options(${target} PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
    endforeach()
endif()