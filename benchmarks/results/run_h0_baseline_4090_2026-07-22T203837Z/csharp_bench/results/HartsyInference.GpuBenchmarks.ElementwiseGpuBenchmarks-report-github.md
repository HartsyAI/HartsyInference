```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-GDLSTF : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method       | SizeIndex | Mean      | Error      | StdDev    |
|------------- |---------- |----------:|-----------:|----------:|
| **Silu**         | **0**         | **121.86 μs** | **238.927 μs** | **62.049 μs** |
| Gelu         | 0         | 115.64 μs | 240.489 μs | 62.454 μs |
| BroadcastAdd | 0         | 132.03 μs | 546.779 μs | 84.615 μs |
| **Silu**         | **1**         |  **68.25 μs** | **196.521 μs** | **30.412 μs** |
| Gelu         | 1         |  35.26 μs |  81.279 μs | 12.578 μs |
| BroadcastAdd | 1         |  38.26 μs |   7.007 μs |  1.084 μs |
| **Silu**         | **2**         | **144.82 μs** | **309.136 μs** | **80.282 μs** |
| Gelu         | 2         | 118.21 μs | 376.681 μs | 58.292 μs |
| BroadcastAdd | 2         |  49.98 μs |   6.723 μs |  1.746 μs |
| **Silu**         | **3**         |  **42.09 μs** |  **20.297 μs** |  **5.271 μs** |
| Gelu         | 3         |  61.09 μs | 102.012 μs | 26.492 μs |
| BroadcastAdd | 3         |  36.37 μs |   5.774 μs |  0.893 μs |
