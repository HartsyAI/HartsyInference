```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-ZXSDEB : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method  | ShapeIndex | Mean       | Error      | StdDev    |
|-------- |----------- |-----------:|-----------:|----------:|
| **RmsNorm** | **0**          |   **727.0 μs** |   **512.4 μs** |  **79.30 μs** |
| **RmsNorm** | **1**          | **1,454.8 μs** |   **850.7 μs** | **131.65 μs** |
| **RmsNorm** | **2**          | **1,763.7 μs** |   **603.0 μs** |  **93.32 μs** |
| **RmsNorm** | **3**          | **1,267.6 μs** | **1,135.4 μs** | **294.86 μs** |
| **RmsNorm** | **4**          | **2,130.9 μs** | **1,160.7 μs** | **301.44 μs** |
