
BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-FXWCEQ : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

 Method              | ShapeIndex | Mean        | Error      | StdDev    |
-------------------- |----------- |------------:|-----------:|----------:|
 **GroupNorm**           | **0**          |  **2,682.7 μs** |   **957.8 μs** | **148.22 μs** |
 GroupNormSilu_Fused | 0          |  2,836.9 μs | 1,098.4 μs | 285.24 μs |
 **GroupNorm**           | **1**          |  **1,278.0 μs** |   **484.5 μs** | **125.82 μs** |
 GroupNormSilu_Fused | 1          |  1,311.8 μs |   565.2 μs | 146.79 μs |
 **GroupNorm**           | **2**          |    **729.8 μs** |   **585.6 μs** | **152.09 μs** |
 GroupNormSilu_Fused | 2          |    706.1 μs |   489.2 μs |  75.71 μs |
 **GroupNorm**           | **3**          |  **4,533.7 μs** |   **271.9 μs** |  **42.07 μs** |
 GroupNormSilu_Fused | 3          |  4,514.3 μs |   247.5 μs |  64.27 μs |
 **GroupNorm**           | **4**          | **40,785.1 μs** | **2,167.9 μs** | **335.49 μs** |
 GroupNormSilu_Fused | 4          | 41,057.7 μs |   408.8 μs | 106.16 μs |
