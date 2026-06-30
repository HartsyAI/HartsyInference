```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-ZXSDEB : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method    | ShapeIndex | Mean       | Error      | StdDev    |
|---------- |----------- |-----------:|-----------:|----------:|
| **LayerNorm** | **0**          | **2,644.4 μs** | **1,064.0 μs** | **276.32 μs** |
| **LayerNorm** | **1**          |   **872.9 μs** |   **687.1 μs** | **178.44 μs** |
| **LayerNorm** | **2**          | **1,529.5 μs** |   **438.0 μs** |  **67.77 μs** |
| **LayerNorm** | **3**          | **2,408.8 μs** |   **961.7 μs** | **249.76 μs** |
| **LayerNorm** | **4**          | **1,177.0 μs** |   **781.2 μs** | **202.87 μs** |
