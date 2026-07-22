```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-GDLSTF : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method          | M    | Mean       | Error     | StdDev    | Ratio | RatioSD |
|---------------- |----- |-----------:|----------:|----------:|------:|--------:|
| **Linear_F16**      | **1**    |   **152.4 μs** | **153.13 μs** |  **39.77 μs** |  **1.05** |    **0.34** |
| QuantMatMul_Q4K | 1    |   191.4 μs |   7.44 μs |   1.15 μs |  1.32 |    0.29 |
|                 |      |            |           |           |       |         |
| **Linear_F16**      | **1024** |   **746.2 μs** | **650.96 μs** | **169.05 μs** |  **1.04** |    **0.31** |
| QuantMatMul_Q4K | 1024 |   810.1 μs | 295.40 μs |  76.71 μs |  1.13 |    0.25 |
|                 |      |            |           |           |       |         |
| **Linear_F16**      | **4096** | **2,400.1 μs** | **154.46 μs** |  **40.11 μs** |  **1.00** |    **0.02** |
| QuantMatMul_Q4K | 4096 | 2,459.5 μs | 351.04 μs |  91.16 μs |  1.02 |    0.04 |
