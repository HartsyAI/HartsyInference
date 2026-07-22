```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-DGGSIS : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method          | SizeIndex | Mean     | Error    | StdDev    | Median   |
|---------------- |---------- |---------:|---------:|----------:|---------:|
| **AllocFree_Sync**  | **0**         | **20.69 μs** | **30.36 μs** |  **7.885 μs** | **25.33 μs** |
| AllocFree_Async | 0         | 38.24 μs | 36.94 μs |  9.593 μs | 34.67 μs |
| **AllocFree_Sync**  | **1**         | **19.40 μs** | **28.11 μs** |  **7.300 μs** | **18.98 μs** |
| AllocFree_Async | 1         | 30.71 μs | 20.26 μs |  3.135 μs | 31.71 μs |
| **AllocFree_Sync**  | **2**         | **14.35 μs** | **11.81 μs** |  **1.827 μs** | **14.04 μs** |
| AllocFree_Async | 2         | 30.69 μs | 56.97 μs | 14.794 μs | 23.33 μs |
| **AllocFree_Sync**  | **3**         | **18.10 μs** | **30.51 μs** |  **7.924 μs** | **19.94 μs** |
| AllocFree_Async | 3         | 42.38 μs | 12.53 μs |  3.254 μs | 42.17 μs |
| **AllocFree_Sync**  | **4**         | **18.93 μs** | **30.56 μs** |  **7.937 μs** | **13.77 μs** |
| AllocFree_Async | 4         | 32.84 μs | 29.94 μs |  4.634 μs | 32.66 μs |
