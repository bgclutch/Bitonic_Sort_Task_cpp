find_package(OpenCL REQUIRED)
add_executable(bitonic_sort)

target_compile_definitions(bitonic_sort PRIVATE
    KERNELS_ABS_PATH="${CMAKE_SOURCE_DIR}/kernels/"
)
target_compile_definitions(bitonic_sort PRIVATE
    CL_TARGET_OPENCL_VERSION=200
)
target_compile_definitions(bitonic_sort PRIVATE
    CL_HPP_TARGET_OPENCL_VERSION=200
)

target_link_libraries(bitonic_sort OpenCL::OpenCL)

target_include_directories(bitonic_sort PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${OpenCL_INCLUDE_DIRS}
)

target_sources(bitonic_sort PRIVATE
    src/main.cpp
)
