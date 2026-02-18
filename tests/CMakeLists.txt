find_package(GTest REQUIRED)
find_package(OpenCL REQUIRED)

add_executable(tests
    Unit/test.cpp
    ${CMAKE_SOURCE_DIR}/bitonic_sort/src/utils.cpp
)

target_compile_features(tests PUBLIC cxx_std_23)
target_link_libraries(tests PRIVATE OpenCL::OpenCL)
target_compile_definitions(tests PRIVATE
    KERNELS_ABS_PATH="${CMAKE_SOURCE_DIR}/kernels/"
)

target_compile_definitions(tests PRIVATE
    CL_TARGET_OPENCL_VERSION=200
    CL_HPP_TARGET_OPENCL_VERSION=200
)

target_include_directories(tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/bitonic_sort/inc
    ${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    OpenCL::OpenCL
)

add_test(NAME unit_test COMMAND tests)
