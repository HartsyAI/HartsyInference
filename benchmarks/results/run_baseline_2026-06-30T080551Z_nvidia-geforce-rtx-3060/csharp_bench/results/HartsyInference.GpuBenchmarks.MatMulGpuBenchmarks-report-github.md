```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-ZXSDEB : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method     | ShapeIndex | Mean      | Error     | StdDev    |
|----------- |----------- |----------:|----------:|----------:|
| **MatMul_F32** | **0**          |  **4.871 ms** | **0.5069 ms** | **0.0784 ms** |
| MatMul_F16 | 0          |  2.290 ms | 0.5751 ms | 0.0890 ms |
| **MatMul_F32** | **1**          |  **5.815 ms** | **2.1191 ms** | **0.5503 ms** |
| MatMul_F16 | 1          |  2.199 ms | 0.3582 ms | 0.0930 ms |
| **MatMul_F32** | **2**          | **21.317 ms** | **0.4793 ms** | **0.1245 ms** |
| MatMul_F16 | 2          |  8.818 ms | 0.7335 ms | 0.1905 ms |
| **MatMul_F32** | **3**          | **21.984 ms** | **3.9461 ms** | **1.0248 ms** |
| MatMul_F16 | 3          |  8.696 ms | 0.2743 ms | 0.0712 ms |
| **MatMul_F32** | **4**          | **23.296 ms** | **1.2199 ms** | **0.3168 ms** |
| MatMul_F16 | 4          | 10.506 ms | 1.0895 ms | 0.2829 ms |
| **MatMul_F32** | **5**          | **30.260 ms** | **1.5035 ms** | **0.3905 ms** |
| MatMul_F16 | 5          | 13.782 ms | 0.6261 ms | 0.0969 ms |
| **MatMul_F32** | **6**          |  **6.016 ms** | **0.5106 ms** | **0.0790 ms** |
| MatMul_F16 | 6          |  2.588 ms | 0.5618 ms | 0.0869 ms |
| **MatMul_F32** | **7**          | **35.606 ms** | **1.3221 ms** | **0.3433 ms** |
| MatMul_F16 | 7          | 15.704 ms | 0.9737 ms | 0.2529 ms |
| **MatMul_F32** | **8**          | **13.256 ms** | **1.3087 ms** | **0.2025 ms** |
| MatMul_F16 | 8          |  5.963 ms | 0.4656 ms | 0.0721 ms |
| **MatMul_F32** | **9**          | **23.062 ms** | **1.2043 ms** | **0.3127 ms** |
| MatMul_F16 | 9          |  9.809 ms | 0.6010 ms | 0.1561 ms |
