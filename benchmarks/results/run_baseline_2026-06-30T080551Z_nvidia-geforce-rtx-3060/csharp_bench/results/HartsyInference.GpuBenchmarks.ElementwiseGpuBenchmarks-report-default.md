
BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-ZXSDEB : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

 Method       | SizeIndex | Mean        | Error        | StdDev     |
------------- |---------- |------------:|-------------:|-----------:|
 **Silu**         | **0**         | **2,959.08 μs** | **1,075.832 μs** | **279.390 μs** |
 Gelu         | 0         | 2,679.79 μs |   882.057 μs | 229.067 μs |
 BroadcastAdd | 0         |   168.41 μs |     5.984 μs |   1.554 μs |
 **Silu**         | **1**         |   **609.53 μs** |   **204.482 μs** |  **31.644 μs** |
 Gelu         | 1         |   732.45 μs |   872.612 μs | 135.038 μs |
 BroadcastAdd | 1         |    67.21 μs |     6.325 μs |   1.643 μs |
 **Silu**         | **2**         | **2,481.13 μs** |   **511.822 μs** |  **79.205 μs** |
 Gelu         | 2         | 2,477.20 μs |   478.584 μs |  74.061 μs |
 BroadcastAdd | 2         |   162.69 μs |     5.252 μs |   0.813 μs |
 **Silu**         | **3**         |    **56.79 μs** |   **151.538 μs** |  **23.451 μs** |
 Gelu         | 3         |    52.07 μs |   154.444 μs |  23.900 μs |
 BroadcastAdd | 3         |    34.67 μs |    30.037 μs |   4.648 μs |
