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
| 8                    | 674.688 ms      | 512.027 ms      | 161.445 ms    | 0.62 ms    | 825.85                  | 1086.6                  |
| 256                  | 730.469 ms      | 576.047 ms      | 154.192 ms    | 13.917 ms  | 41.3                    | 52.5                    |
| 1024                 | 705.619 ms      | 544.997 ms      | 160.405 ms    | 62.756 ms  | 8.68                    | 11.24                   |
| 97393                | 6119.47 ms      | 5326.31 ms      | 792.858 ms    | 11236 ms   | 0,474                   | 0.544                   |
| 528323               | 11416.2 ms      | 13927.2 ms      | 2510.76 ms    | 53646 ms   | 0.212                   | 0.26                    |
| 10000000             | 411024  ms      | 378665  ms      | 32358.3 ms    | 1312900 ms | 0.29                    | 0.313                   |



**Naive bitonic sort GPU kernel and mine bitonic::bitonic_sort on CPU**
| Elements amount| GPU Total time (Wall time) | Kernel Execution time | Data Transfer time | CPU time |
|-----------------------|------------------|------------------|---------------|----------|
| 8                    | 3035.45 ms      | 2896.56 ms      | 138.649 ms    | 0.659 ms    | 
| 256                  | 3465 ms         | 3383.84 ms      | 137.672 ms    | 43.162 ms   | 
| 1024                 | 3776.25 ms      | 3629.01 ms      | 147.009 ms    | 206.639 ms  |
| 97393                | 7322.94 ms      | 6711.05 ms      | 611.625 ms    | 37820.1 ms  | 
| 528323               | 18614.8 ms      | 16490.3 ms      | 2124.3 ms     | 335343 ms   |
| 10000000             | 497106 ms       | 462788 ms       | 34317 ms      | 7202900 ms  |
