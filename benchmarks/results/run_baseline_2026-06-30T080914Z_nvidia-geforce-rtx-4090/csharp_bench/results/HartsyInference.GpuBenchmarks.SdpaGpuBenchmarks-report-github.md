```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-FXWCEQ : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method   | ShapeIndex | Mean      | Error     | StdDev    |
|--------- |----------- |----------:|----------:|----------:|
| **Sdpa_F32** | **0**          |  **2.930 ms** | **0.5310 ms** | **0.0822 ms** |
| Sdpa_F16 | 0          |  1.457 ms | 0.6390 ms | 0.0989 ms |
| **Sdpa_F32** | **1**          | **35.178 ms** | **1.7929 ms** | **0.2775 ms** |
| Sdpa_F16 | 1          | 16.213 ms | 0.4089 ms | 0.0633 ms |
| **Sdpa_F32** | **2**          |  **3.195 ms** | **0.7098 ms** | **0.1843 ms** |
| Sdpa_F16 | 2          |  1.625 ms | 0.5224 ms | 0.1357 ms |
| **Sdpa_F32** | **3**          |  **7.270 ms** | **1.3550 ms** | **0.3519 ms** |
| Sdpa_F16 | 3          |  3.506 ms | 1.4620 ms | 0.3797 ms |
| **Sdpa_F32** | **4**          |  **9.819 ms** | **1.3992 ms** | **0.3634 ms** |
| Sdpa_F16 | 4          |  4.481 ms | 1.2500 ms | 0.3246 ms |
| **Sdpa_F32** | **5**          | **10.452 ms** | **1.6759 ms** | **0.4352 ms** |
| Sdpa_F16 | 5          |  4.694 ms | 1.2520 ms | 0.3251 ms |
| **Sdpa_F32** | **6**          |  **5.796 ms** | **0.4077 ms** | **0.0631 ms** |
| Sdpa_F16 | 6          |  2.638 ms | 2.5928 ms | 0.6733 ms |
| **Sdpa_F32** | **7**          |  **2.889 ms** | **0.7953 ms** | **0.1231 ms** |
| Sdpa_F16 | 7          |  1.754 ms | 1.0960 ms | 0.2846 ms |
