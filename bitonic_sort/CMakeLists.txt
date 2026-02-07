add_executable(bitonic_sort)

target_include_directories(bitonic_sort PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_sources(bitonic_sort PRIVATE
    src/main.cpp
)
