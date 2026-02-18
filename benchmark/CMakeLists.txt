add_executable(benchmark)

target_include_directories(benchmark PRIVATE ${PROJECT_SOURCE_DIR}/bitonic_sort/inc)

target_compile_features(benchmark PUBLIC cxx_std_23)

target_compile_options(benchmark PRIVATE
    -O2
)

target_sources(benchmark PRIVATE
    benchmark.cpp
)