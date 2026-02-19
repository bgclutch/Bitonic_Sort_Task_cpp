# Bitonic sort
This project provides an implementation of a Bitonic sort algorithm in C++ and OpenCL.  
It also includes automated tests comparing the results of the Bitonic sort with the C++ Standard Library `std::sort` and mine CPU-based `bitonic::bitonic_sort`.

## Features:
1. An implementation of a Bitonic sort algorithm with OpenCL library
2. Comparison of results with `std::sort` and CPU-based `bitonic::bitonic_sort` for correctness
3. Python scripts for automated testing and output verification

## Installation:
Clone this repository, then reach the project directory:
```sh
git clone git@github.com:bgclutch/Bitonic_Sort_Task_cpp.git
cd Bitonic_Sort_Task_cpp
```

## Building:
1. Build the project:
 ```sh
cmake -B build
cmake --build build
```

## Usage:
1. Navigate to the ```build``` folder:
```sh
cd build
```
2. Choose tree to run:
```sh
./bitonic_sort/bitonic_sort
```

## Running tests:
For End To End tests:
1.1 Navigate to the ```tests``` directory:
```sh
cd tests/End_To_End
```
1.2 Run default tests with the Python script:
```sh
python3 testrun.py
```
1.3 (Optional) Or ```regenerate``` test cases:
```sh
python3 testgen.py
```
And run it as in step 2.

For unit tests:
2.1 Navigate to the ```build``` folder:
```sh
cd build
```
2. Run unit tests:
```sh
./tests/tests
```
## Benchmark run
1. To build the project in benchmark mode:
```sh
cmake -DENABLE_BECHMARK=ON -B build
cmake --build build
```
2.1 Run benchmark with default data:
```sh
./build/benchmark/benchmark
```
2.2 Or use your data: 
```sh
./build/benchmark/benchmark "USER'S FILE"
```

## Benchmark results
Benchmark for 6 benchmark tests with different data set size in each,
using -O2 optimisation

- Device: Huawei MateBook XPro 2022
- CPU: Intel Core i7 1260-P
- Memory: 16 GB Unified Memory
- Graphics: Intel Iris Xe Graphics

**Fast bitonic sort GPU kernel and std::sort** 
  
| Elements amount| GPU Total time (Wall time) | Kernel Execution time | Data Transfer time | CPU time | Kernel time to CPU time ratio | Wall time to CPU time ratio |
|-----------------------|------------------|------------------|---------------|----------|-------------------------|------------------------|
| 8                    | 674.688 us      | 512.027 us      | 161.445 us    | 0.62 us    | 825.85                  | 1086.6                  |
| 256                  | 730.469 us      | 576.047 us      | 154.192 us    | 13.917 us  | 41.3                    | 52.5                    |
| 1024                 | 705.619 us      | 544.997 us      | 160.405 us    | 62.756 us  | 8.68                    | 11.24                   |
| 97393                | 6119.47 us      | 5326.31 us      | 792.858 us    | 11236 us   | 0,474                   | 0.544                   |
| 528323               | 11416.2 us      | 13927.2 us      | 2510.76 us    | 53646 us   | 0.212                   | 0.26                    |
| 10000000             | 411024  us      | 378665  us      | 32358.3 us    | 1312900 us | 0.29                    | 0.313                   |



**Naive bitonic sort GPU kernel and mine bitonic::bitonic_sort on CPU**
| Elements amount| GPU Total time (Wall time) | Kernel Execution time | Data Transfer time | CPU time |
|-----------------------|------------------|------------------|---------------|----------|
| 8                    | 3035.45 us      | 2896.56 us      | 138.649 us    | 0.659 us    | 
| 256                  | 3465 us         | 3383.84 us      | 137.672 us    | 43.162 us   | 
| 1024                 | 3776.25 us      | 3629.01 us      | 147.009 us    | 206.639 us  |
| 97393                | 7322.94 us      | 6711.05 us      | 611.625 us    | 37820.1 us  | 
| 528323               | 18614.8 us      | 16490.3 us      | 2124.3 us     | 335343 us   |
| 10000000             | 497106 us       | 462788 us       | 34317 us      | 7202900 us  |
