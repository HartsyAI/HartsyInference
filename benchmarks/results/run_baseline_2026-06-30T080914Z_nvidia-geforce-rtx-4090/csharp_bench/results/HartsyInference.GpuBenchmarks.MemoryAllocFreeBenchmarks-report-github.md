```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-FXWCEQ : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method          | SizeIndex | Mean        | Error       | StdDev    |
|---------------- |---------- |------------:|------------:|----------:|
| **AllocFree_Sync**  | **0**         |    **187.7 μs** |    **28.28 μs** |   **7.34 μs** |
| AllocFree_Async | 0         |    184.6 μs |    12.27 μs |   1.90 μs |
| **AllocFree_Sync**  | **1**         |    **181.6 μs** |    **20.32 μs** |   **5.28 μs** |
| AllocFree_Async | 1         |    184.6 μs |    22.30 μs |   5.79 μs |
| **AllocFree_Sync**  | **2**         |    **884.0 μs** |    **53.98 μs** |   **8.35 μs** |
| AllocFree_Async | 2         |    875.8 μs |    20.09 μs |   5.22 μs |
| **AllocFree_Sync**  | **3**         |  **4,992.3 μs** |   **175.05 μs** |  **45.46 μs** |
| AllocFree_Async | 3         |  4,960.0 μs |   131.17 μs |  34.06 μs |
| **AllocFree_Sync**  | **4**         | **21,352.8 μs** | **2,016.39 μs** | **523.65 μs** |
| AllocFree_Async | 4         | 20,258.7 μs |   387.08 μs | 100.52 μs |
