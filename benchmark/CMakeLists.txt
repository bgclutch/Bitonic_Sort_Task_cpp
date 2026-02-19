add_executable(benchmark)

target_compile_features(benchmark PUBLIC cxx_std_23)
target_link_libraries(benchmark PRIVATE OpenCL::OpenCL)

target_include_directories(benchmark PRIVATE
    ${CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/bitonic_sort/inc
    ${OpenCL_INCLUDE_DIRS}
)

target_compile_definitions(benchmark PRIVATE
    CL_TARGET_OPENCL_VERSION=200
    CL_HPP_TARGET_OPENCL_VERSION=200
)

target_compile_options(benchmark PRIVATE
    -O2
)

target_sources(benchmark PRIVATE
    benchmark.cpp
    ${PROJECT_SOURCE_DIR}/bitonic_sort/src/utils.cpp
)