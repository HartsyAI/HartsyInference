
BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-ZXSDEB : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

 Method              | ShapeIndex | Mean        | Error      | StdDev    |
-------------------- |----------- |------------:|-----------:|----------:|
 **GroupNorm**           | **0**          |  **3,083.6 μs** |   **247.6 μs** |  **64.30 μs** |
 GroupNormSilu_Fused | 0          |  3,250.5 μs |   472.1 μs | 122.62 μs |
 **GroupNorm**           | **1**          |  **1,469.2 μs** |   **420.0 μs** | **109.09 μs** |
 GroupNormSilu_Fused | 1          |  1,732.2 μs |   614.1 μs | 159.47 μs |
 **GroupNorm**           | **2**          |    **844.9 μs** |   **544.2 μs** | **141.34 μs** |
 GroupNormSilu_Fused | 2          |    857.2 μs |   459.2 μs |  71.06 μs |
 **GroupNorm**           | **3**          |  **5,644.1 μs** |   **299.1 μs** |  **77.67 μs** |
 GroupNormSilu_Fused | 3          |  5,292.5 μs |   642.1 μs |  99.36 μs |
 **GroupNorm**           | **4**          | **45,123.7 μs** | **2,808.4 μs** | **729.34 μs** |
 GroupNormSilu_Fused | 4          | 45,458.9 μs | 3,044.1 μs | 790.54 μs |
