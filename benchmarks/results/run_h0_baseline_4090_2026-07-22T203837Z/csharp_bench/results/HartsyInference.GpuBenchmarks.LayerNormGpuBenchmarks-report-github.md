```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-GDLSTF : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method    | ShapeIndex | Mean      | Error       | StdDev    | Median    |
|---------- |----------- |----------:|------------:|----------:|----------:|
| **LayerNorm** | **0**          | **129.68 μs** |   **329.33 μs** |  **50.96 μs** | **133.32 μs** |
| **LayerNorm** | **1**          |  **82.89 μs** |   **157.09 μs** |  **40.80 μs** |  **76.15 μs** |
| **LayerNorm** | **2**          | **105.41 μs** |   **252.00 μs** |  **65.44 μs** |  **62.46 μs** |
| **LayerNorm** | **3**          | **798.49 μs** | **3,092.66 μs** | **803.15 μs** | **568.26 μs** |
| **LayerNorm** | **4**          |  **93.72 μs** |   **237.55 μs** |  **61.69 μs** |  **53.17 μs** |
