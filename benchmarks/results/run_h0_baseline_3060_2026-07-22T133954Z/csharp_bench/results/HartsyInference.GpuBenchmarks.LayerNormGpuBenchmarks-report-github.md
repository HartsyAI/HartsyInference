```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-DGGSIS : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method    | ShapeIndex | Mean     | Error    | StdDev    | Median   |
|---------- |----------- |---------:|---------:|----------:|---------:|
| **LayerNorm** | **0**          | **357.8 μs** | **430.9 μs** | **111.89 μs** | **380.2 μs** |
| **LayerNorm** | **1**          | **185.0 μs** | **394.1 μs** | **102.35 μs** | **116.6 μs** |
| **LayerNorm** | **2**          | **277.9 μs** | **415.7 μs** | **107.96 μs** | **212.8 μs** |
| **LayerNorm** | **3**          | **381.2 μs** | **488.6 μs** | **126.90 μs** | **417.9 μs** |
| **LayerNorm** | **4**          | **170.5 μs** | **211.5 μs** |  **32.74 μs** | **159.7 μs** |
