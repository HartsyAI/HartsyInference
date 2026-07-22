```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-GDLSTF : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method               | KvLen | Mean      | Error     | StdDev    |
|--------------------- |------ |----------:|----------:|----------:|
| **FlashAttn_Decode_F32** | **512**   |  **87.72 μs** | **91.931 μs** | **23.874 μs** |
| **FlashAttn_Decode_F32** | **2048**  |  **88.24 μs** | **26.387 μs** |  **6.853 μs** |
| **FlashAttn_Decode_F32** | **8192**  | **189.84 μs** |  **9.232 μs** |  **1.429 μs** |
