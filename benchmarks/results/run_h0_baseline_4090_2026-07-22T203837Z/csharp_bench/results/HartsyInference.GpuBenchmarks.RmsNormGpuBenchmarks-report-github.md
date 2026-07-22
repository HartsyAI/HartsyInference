```

BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-GDLSTF : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

```
| Method  | ShapeIndex | Mean      | Error     | StdDev   | Median    |
|-------- |----------- |----------:|----------:|---------:|----------:|
| **RmsNorm** | **0**          |  **55.26 μs** | **120.46 μs** | **18.64 μs** |  **49.42 μs** |
| **RmsNorm** | **1**          |  **90.00 μs** | **317.18 μs** | **49.08 μs** |  **83.44 μs** |
| **RmsNorm** | **2**          | **168.95 μs** | **344.92 μs** | **89.57 μs** | **155.34 μs** |
| **RmsNorm** | **3**          |  **98.49 μs** | **318.29 μs** | **82.66 μs** |  **43.46 μs** |
| **RmsNorm** | **4**          |  **97.19 μs** | **231.09 μs** | **60.01 μs** |  **59.39 μs** |
