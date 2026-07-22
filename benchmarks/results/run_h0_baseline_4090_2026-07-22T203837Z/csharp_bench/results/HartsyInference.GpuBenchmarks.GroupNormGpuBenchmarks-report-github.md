```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-GDLSTF : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method              | ShapeIndex | Mean       | Error      | StdDev    | Median     |
|-------------------- |----------- |-----------:|-----------:|----------:|-----------:|
| **GroupNorm**           | **0**          |   **366.6 μs** |   **318.1 μs** |  **49.22 μs** |   **362.2 μs** |
| GroupNormSilu_Fused | 0          |   403.2 μs |   178.4 μs |  46.33 μs |   397.1 μs |
| **GroupNorm**           | **1**          |   **222.4 μs** |   **203.4 μs** |  **52.82 μs** |   **212.9 μs** |
| GroupNormSilu_Fused | 1          |   252.8 μs |   184.5 μs |  47.93 μs |   265.7 μs |
| **GroupNorm**           | **2**          |   **145.1 μs** |   **223.6 μs** |  **58.06 μs** |   **104.8 μs** |
| GroupNormSilu_Fused | 2          |   131.8 μs |   180.0 μs |  27.86 μs |   120.4 μs |
| **GroupNorm**           | **3**          |   **614.5 μs** |   **148.5 μs** |  **22.98 μs** |   **616.7 μs** |
| GroupNormSilu_Fused | 3          |   761.6 μs |   447.3 μs | 116.16 μs |   748.6 μs |
| **GroupNorm**           | **4**          | **7,775.2 μs** | **1,541.0 μs** | **400.18 μs** | **7,579.9 μs** |
| GroupNormSilu_Fused | 4          | 7,859.5 μs | 1,030.9 μs | 159.54 μs | 7,913.2 μs |
