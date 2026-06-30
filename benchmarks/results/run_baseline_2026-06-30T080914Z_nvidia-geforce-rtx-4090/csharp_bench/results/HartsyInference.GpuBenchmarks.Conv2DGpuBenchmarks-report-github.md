```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-FXWCEQ : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method     | ShapeIndex | Mean       | Error     | StdDev    |
|----------- |----------- |-----------:|----------:|----------:|
| **Conv2D_F32** | **0**          |   **7.465 ms** | **1.8757 ms** | **0.2903 ms** |
| **Conv2D_F32** | **1**          |   **5.204 ms** | **2.1474 ms** | **0.3323 ms** |
| **Conv2D_F32** | **2**          |   **8.956 ms** | **1.7914 ms** | **0.2772 ms** |
| **Conv2D_F32** | **3**          |   **3.637 ms** | **1.3415 ms** | **0.3484 ms** |
| **Conv2D_F32** | **4**          |   **2.818 ms** | **0.3894 ms** | **0.0603 ms** |
| **Conv2D_F32** | **5**          |  **11.628 ms** | **0.2893 ms** | **0.0448 ms** |
| **Conv2D_F32** | **6**          |  **92.028 ms** | **1.6132 ms** | **0.4189 ms** |
| **Conv2D_F32** | **7**          | **175.349 ms** | **1.0021 ms** | **0.1551 ms** |
