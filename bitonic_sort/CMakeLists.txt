find_package(OpenCL REQUIRED)
add_executable(bitonic_sort)

target_compile_features(bitonic_sort PUBLIC cxx_std_23)
target_link_libraries(bitonic_sort PRIVATE OpenCL::OpenCL)

target_compile_definitions(bitonic_sort PRIVATE
    KERNELS_ABS_PATH="${CMAKE_SOURCE_DIR}/kernels/"
    CL_TARGET_OPENCL_VERSION=200
    CL_HPP_TARGET_OPENCL_VERSION=200
)

target_include_directories(bitonic_sort PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${OpenCL_INCLUDE_DIRS}
)

target_sources(bitonic_sort PRIVATE
    src/main.cpp
    src/utils.cpp
)
