```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-DGGSIS : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method     | ShapeIndex | Mean      | Error     | StdDev    |
|----------- |----------- |----------:|----------:|----------:|
| **Conv2D_F32** | **0**          |  **5.670 ms** | **0.1824 ms** | **0.0474 ms** |
| **Conv2D_F32** | **1**          |  **3.999 ms** | **0.2353 ms** | **0.0611 ms** |
| **Conv2D_F32** | **2**          |  **4.022 ms** | **0.0268 ms** | **0.0070 ms** |
| **Conv2D_F32** | **3**          |  **1.499 ms** | **0.0258 ms** | **0.0040 ms** |
| **Conv2D_F32** | **4**          |  **1.133 ms** | **0.1067 ms** | **0.0277 ms** |
| **Conv2D_F32** | **5**          |  **6.110 ms** | **0.1354 ms** | **0.0352 ms** |
| **Conv2D_F32** | **6**          | **43.299 ms** | **6.6854 ms** | **1.7362 ms** |
| **Conv2D_F32** | **7**          | **47.855 ms** | **0.7405 ms** | **0.1923 ms** |
