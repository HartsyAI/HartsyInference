```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-DGGSIS : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method  | ShapeIndex | Mean     | Error    | StdDev    | Median   |
|-------- |----------- |---------:|---------:|----------:|---------:|
| **RmsNorm** | **0**          | **146.6 μs** | **212.6 μs** |  **32.90 μs** | **142.9 μs** |
| **RmsNorm** | **1**          | **352.1 μs** | **698.0 μs** | **181.27 μs** | **230.3 μs** |
| **RmsNorm** | **2**          | **368.3 μs** | **464.1 μs** | **120.53 μs** | **415.2 μs** |
| **RmsNorm** | **3**          | **257.8 μs** | **594.3 μs** | **154.34 μs** | **189.4 μs** |
| **RmsNorm** | **4**          | **313.4 μs** | **409.4 μs** | **106.32 μs** | **269.0 μs** |
