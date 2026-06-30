```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-FXWCEQ : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method     | ShapeIndex | Mean      | Error     | StdDev    |
|----------- |----------- |----------:|----------:|----------:|
| **MatMul_F32** | **0**          |  **3.399 ms** | **0.4011 ms** | **0.0621 ms** |
| MatMul_F16 | 0          |  1.623 ms | 0.7041 ms | 0.1829 ms |
| **MatMul_F32** | **1**          |  **3.404 ms** | **0.5673 ms** | **0.0878 ms** |
| MatMul_F16 | 1          |  1.503 ms | 0.4142 ms | 0.0641 ms |
| **MatMul_F32** | **2**          | **11.518 ms** | **1.7724 ms** | **0.4603 ms** |
| MatMul_F16 | 2          |  5.231 ms | 0.7898 ms | 0.1222 ms |
| **MatMul_F32** | **3**          | **14.768 ms** | **1.1544 ms** | **0.2998 ms** |
| MatMul_F16 | 3          |  6.598 ms | 0.3537 ms | 0.0547 ms |
| **MatMul_F32** | **4**          | **16.775 ms** | **0.5059 ms** | **0.1314 ms** |
| MatMul_F16 | 4          |  7.510 ms | 0.5566 ms | 0.0861 ms |
| **MatMul_F32** | **5**          | **21.169 ms** | **0.7852 ms** | **0.2039 ms** |
| MatMul_F16 | 5          | 10.375 ms | 0.9482 ms | 0.2462 ms |
| **MatMul_F32** | **6**          |  **4.492 ms** | **0.5462 ms** | **0.0845 ms** |
| MatMul_F16 | 6          |  1.989 ms | 0.3478 ms | 0.0538 ms |
| **MatMul_F32** | **7**          | **25.508 ms** | **1.1595 ms** | **0.1794 ms** |
| MatMul_F16 | 7          | 11.886 ms | 0.8754 ms | 0.2273 ms |
| **MatMul_F32** | **8**          |  **9.514 ms** | **0.7848 ms** | **0.2038 ms** |
| MatMul_F16 | 8          |  4.707 ms | 0.9769 ms | 0.2537 ms |
| **MatMul_F32** | **9**          | **16.411 ms** | **0.7898 ms** | **0.2051 ms** |
| MatMul_F16 | 9          |  7.553 ms | 0.7116 ms | 0.1848 ms |
