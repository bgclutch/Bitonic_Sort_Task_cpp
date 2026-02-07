cmake_minimum_required(VERSION 3.13)

project(bitonic_sort)
add_subdirectory(bitonic_sort)

option(ENABLE_ASAN OFF)

target_compile_features(bitonic_sort PUBLIC cxx_std_23)

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
    target_compile_options(bitonic_sort PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
    target_link_options(bitonic_sort PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
endif()