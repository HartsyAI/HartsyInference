```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-FXWCEQ : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method       | SizeIndex | Mean        | Error        | StdDev     |
|------------- |---------- |------------:|-------------:|-----------:|
| **Silu**         | **0**         | **2,493.31 μs** |   **671.421 μs** | **174.366 μs** |
| Gelu         | 0         | 2,435.89 μs | 1,375.465 μs | 357.204 μs |
| BroadcastAdd | 0         |    51.86 μs |    31.800 μs |   8.258 μs |
| **Silu**         | **1**         |   **567.08 μs** |   **179.763 μs** |  **27.819 μs** |
| Gelu         | 1         |   593.93 μs |   336.969 μs |  52.146 μs |
| BroadcastAdd | 1         |    39.32 μs |    32.277 μs |   8.382 μs |
| **Silu**         | **2**         | **2,402.57 μs** |   **472.650 μs** |  **73.143 μs** |
| Gelu         | 2         | 2,405.54 μs |   415.912 μs |  64.363 μs |
| BroadcastAdd | 2         |    44.43 μs |     5.464 μs |   0.846 μs |
| **Silu**         | **3**         |    **51.86 μs** |   **153.613 μs** |  **23.772 μs** |
| Gelu         | 3         |    46.59 μs |   133.500 μs |  20.659 μs |
| BroadcastAdd | 3         |    38.75 μs |    30.272 μs |   7.862 μs |
