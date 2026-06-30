```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-FXWCEQ : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method    | ShapeIndex | Mean       | Error    | StdDev    |
|---------- |----------- |-----------:|---------:|----------:|
| **LayerNorm** | **0**          | **2,371.4 μs** | **700.4 μs** | **108.38 μs** |
| **LayerNorm** | **1**          |   **704.0 μs** | **510.4 μs** |  **78.99 μs** |
| **LayerNorm** | **2**          | **1,373.3 μs** | **505.4 μs** |  **78.22 μs** |
| **LayerNorm** | **3**          | **1,670.9 μs** | **451.4 μs** |  **69.85 μs** |
| **LayerNorm** | **4**          | **1,107.9 μs** | **697.6 μs** | **181.17 μs** |
