
BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-DGGSIS : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

 Method       | SizeIndex | Mean      | Error      | StdDev    |
------------- |---------- |----------:|-----------:|----------:|
 **Silu**         | **0**         | **216.87 μs** | **230.916 μs** | **59.968 μs** |
 Gelu         | 0         | 252.11 μs | 268.352 μs | 69.690 μs |
 BroadcastAdd | 0         | 167.52 μs |   9.716 μs |  2.523 μs |
 **Silu**         | **1**         |  **60.09 μs** |  **57.581 μs** |  **8.911 μs** |
 Gelu         | 1         |  55.88 μs |  22.452 μs |  3.475 μs |
 BroadcastAdd | 1         |  66.93 μs |   6.123 μs |  1.590 μs |
 **Silu**         | **2**         | **222.15 μs** | **243.693 μs** | **63.286 μs** |
 Gelu         | 2         | 215.03 μs | 245.774 μs | 63.827 μs |
 BroadcastAdd | 2         | 176.06 μs |  51.426 μs |  7.958 μs |
 **Silu**         | **3**         |  **34.56 μs** |   **8.300 μs** |  **2.156 μs** |
 Gelu         | 3         |  36.13 μs |   6.981 μs |  1.813 μs |
 BroadcastAdd | 3         |  34.26 μs |   7.946 μs |  2.063 μs |
