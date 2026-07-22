```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-GDLSTF : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method          | SizeIndex | Mean      | Error     | StdDev    |
|---------------- |---------- |----------:|----------:|----------:|
| **AllocFree_Sync**  | **0**         |  **7.032 μs** | **2.1726 μs** | **0.3362 μs** |
| AllocFree_Async | 0         | 15.304 μs | 4.4237 μs | 1.1488 μs |
| **AllocFree_Sync**  | **1**         | **12.376 μs** | **8.0173 μs** | **2.0821 μs** |
| AllocFree_Async | 1         | 16.625 μs | 4.8240 μs | 1.2528 μs |
| **AllocFree_Sync**  | **2**         |  **7.275 μs** | **2.4341 μs** | **0.6321 μs** |
| AllocFree_Async | 2         | 14.096 μs | 2.8645 μs | 0.7439 μs |
| **AllocFree_Sync**  | **3**         |  **7.343 μs** | **0.8725 μs** | **0.1350 μs** |
| AllocFree_Async | 3         | 13.738 μs | 2.8253 μs | 0.4372 μs |
| **AllocFree_Sync**  | **4**         |  **7.730 μs** | **1.8908 μs** | **0.4910 μs** |
| AllocFree_Async | 4         | 15.113 μs | 5.8177 μs | 1.5108 μs |
