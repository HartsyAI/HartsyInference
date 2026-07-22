
BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-DGGSIS : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

 Method              | ShapeIndex | Mean        | Error       | StdDev      |
-------------------- |----------- |------------:|------------:|------------:|
 **GroupNorm**           | **0**          |    **813.4 μs** |   **308.32 μs** |    **80.07 μs** |
 GroupNormSilu_Fused | 0          |    861.1 μs |   320.17 μs |    83.15 μs |
 **GroupNorm**           | **1**          |    **444.7 μs** |   **322.49 μs** |    **83.75 μs** |
 GroupNormSilu_Fused | 1          |    453.0 μs |   281.91 μs |    73.21 μs |
 **GroupNorm**           | **2**          |    **333.3 μs** |   **408.61 μs** |   **106.11 μs** |
 GroupNormSilu_Fused | 2          |    320.6 μs |   260.32 μs |    40.29 μs |
 **GroupNorm**           | **3**          |  **1,380.7 μs** |   **114.58 μs** |    **29.76 μs** |
 GroupNormSilu_Fused | 3          |  1,439.0 μs |    88.96 μs |    13.77 μs |
 **GroupNorm**           | **4**          | **10,330.7 μs** | **5,442.41 μs** | **1,413.38 μs** |
 GroupNormSilu_Fused | 4          | 10,302.2 μs | 2,956.63 μs |   767.83 μs |
