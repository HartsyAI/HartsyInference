
BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-GDLSTF : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

 Method     | ShapeIndex | Mean        | Error       | StdDev    |
----------- |----------- |------------:|------------:|----------:|
 **Conv2D_F32** | **0**          |  **1,017.4 μs** |   **312.14 μs** |  **81.06 μs** |
 **Conv2D_F32** | **1**          |    **715.0 μs** |   **255.80 μs** |  **66.43 μs** |
 **Conv2D_F32** | **2**          |    **566.9 μs** |   **170.72 μs** |  **26.42 μs** |
 **Conv2D_F32** | **3**          |    **300.6 μs** |    **28.60 μs** |   **4.43 μs** |
 **Conv2D_F32** | **4**          |    **412.2 μs** |   **262.39 μs** |  **40.61 μs** |
 **Conv2D_F32** | **5**          |  **1,315.2 μs** |   **311.37 μs** |  **80.86 μs** |
 **Conv2D_F32** | **6**          |  **8,568.4 μs** | **1,155.09 μs** | **299.97 μs** |
 **Conv2D_F32** | **7**          | **11,955.8 μs** | **1,753.20 μs** | **455.30 μs** |
