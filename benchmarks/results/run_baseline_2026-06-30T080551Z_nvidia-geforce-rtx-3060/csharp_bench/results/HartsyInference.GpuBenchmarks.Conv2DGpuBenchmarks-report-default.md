
BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-ZXSDEB : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

 Method     | ShapeIndex | Mean       | Error      | StdDev    |
----------- |----------- |-----------:|-----------:|----------:|
 **Conv2D_F32** | **0**          |  **13.972 ms** |  **2.0015 ms** | **0.5198 ms** |
 **Conv2D_F32** | **1**          |  **10.214 ms** |  **1.7263 ms** | **0.4483 ms** |
 **Conv2D_F32** | **2**          |  **14.380 ms** |  **3.9651 ms** | **1.0297 ms** |
 **Conv2D_F32** | **3**          |   **5.287 ms** |  **1.8557 ms** | **0.4819 ms** |
 **Conv2D_F32** | **4**          |   **4.107 ms** |  **0.2312 ms** | **0.0358 ms** |
 **Conv2D_F32** | **5**          |  **19.814 ms** |  **2.1751 ms** | **0.5649 ms** |
 **Conv2D_F32** | **6**          | **173.386 ms** | **10.9507 ms** | **2.8439 ms** |
 **Conv2D_F32** | **7**          | **263.478 ms** | **12.9134 ms** | **1.9984 ms** |
