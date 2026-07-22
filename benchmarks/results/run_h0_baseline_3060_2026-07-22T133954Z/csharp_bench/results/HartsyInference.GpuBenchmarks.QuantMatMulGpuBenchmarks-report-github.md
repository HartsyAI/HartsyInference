```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-DGGSIS : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method          | M    | Mean        | Error     | StdDev    | Ratio | RatioSD |
|---------------- |----- |------------:|----------:|----------:|------:|--------:|
| **Linear_F16**      | **1**    |    **410.9 μs** |  **83.11 μs** |  **21.58 μs** |  **1.00** |    **0.07** |
| QuantMatMul_Q4K | 1    |  1,012.8 μs | 225.31 μs |  58.51 μs |  2.47 |    0.17 |
|                 |      |             |           |           |       |         |
| **Linear_F16**      | **1024** |  **3,726.4 μs** | **743.33 μs** | **115.03 μs** |  **1.00** |    **0.04** |
| QuantMatMul_Q4K | 1024 |  4,279.2 μs | 696.79 μs | 107.83 μs |  1.15 |    0.04 |
|                 |      |             |           |           |       |         |
| **Linear_F16**      | **4096** | **13,565.8 μs** | **770.81 μs** | **200.18 μs** |  **1.00** |    **0.02** |
| QuantMatMul_Q4K | 4096 | 13,915.0 μs | 534.83 μs |  82.76 μs |  1.03 |    0.01 |
