```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-DGGSIS : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method               | KvLen | Mean       | Error     | StdDev   |
|--------------------- |------ |-----------:|----------:|---------:|
| **FlashAttn_Decode_F32** | **512**   |   **300.8 μs** | **113.28 μs** | **29.42 μs** |
| **FlashAttn_Decode_F32** | **2048**  |   **989.6 μs** |  **23.25 μs** |  **3.60 μs** |
| **FlashAttn_Decode_F32** | **8192**  | **3,827.2 μs** |  **64.26 μs** |  **9.94 μs** |
