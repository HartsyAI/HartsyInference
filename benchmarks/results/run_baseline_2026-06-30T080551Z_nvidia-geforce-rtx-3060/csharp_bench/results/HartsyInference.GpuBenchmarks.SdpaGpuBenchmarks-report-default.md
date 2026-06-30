
BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-ZXSDEB : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

 Method   | ShapeIndex | Mean      | Error     | StdDev    |
--------- |----------- |----------:|----------:|----------:|
 **Sdpa_F32** | **0**          |  **4.428 ms** | **0.5656 ms** | **0.1469 ms** |
 Sdpa_F16 | 0          |  2.161 ms | 0.1648 ms | 0.0255 ms |
 **Sdpa_F32** | **1**          | **58.979 ms** | **5.9028 ms** | **1.5329 ms** |
 Sdpa_F16 | 1          | 25.866 ms | 1.1891 ms | 0.3088 ms |
 **Sdpa_F32** | **2**          |  **4.346 ms** | **0.3936 ms** | **0.1022 ms** |
 Sdpa_F16 | 2          |  2.890 ms | 0.6408 ms | 0.1664 ms |
 **Sdpa_F32** | **3**          | **10.632 ms** | **2.1700 ms** | **0.5635 ms** |
 Sdpa_F16 | 3          |  5.807 ms | 2.1405 ms | 0.5559 ms |
 **Sdpa_F32** | **4**          | **13.323 ms** | **1.6437 ms** | **0.4269 ms** |
 Sdpa_F16 | 4          |  6.542 ms | 1.0333 ms | 0.1599 ms |
 **Sdpa_F32** | **5**          | **13.646 ms** | **2.1974 ms** | **0.5706 ms** |
 Sdpa_F16 | 5          |  6.244 ms | 0.3528 ms | 0.0546 ms |
 **Sdpa_F32** | **6**          |  **7.641 ms** | **0.3268 ms** | **0.0849 ms** |
 Sdpa_F16 | 6          |  3.836 ms | 2.1996 ms | 0.5712 ms |
 **Sdpa_F32** | **7**          |  **4.113 ms** | **0.1639 ms** | **0.0426 ms** |
 Sdpa_F16 | 7          |  2.682 ms | 0.4398 ms | 0.1142 ms |
