
BenchmarkDotNet v0.14.0, Linux Mint 22.2 (Zara)
Intel Core i7-6900K CPU 3.20GHz (Skylake), 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.109
  [Host]     : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2
  Job-FXWCEQ : .NET 10.0.9 (10.0.926.27113), X64 RyuJIT AVX2

InvocationCount=1  IterationCount=5  LaunchCount=1  
RunStrategy=Throughput  UnrollFactor=1  WarmupCount=1  

 Method  | ShapeIndex | Mean       | Error    | StdDev    |
-------- |----------- |-----------:|---------:|----------:|
 **RmsNorm** | **0**          |   **695.4 μs** | **441.4 μs** |  **68.31 μs** |
 **RmsNorm** | **1**          | **1,382.5 μs** | **772.6 μs** | **119.56 μs** |
 **RmsNorm** | **2**          | **1,797.1 μs** | **631.5 μs** |  **97.72 μs** |
 **RmsNorm** | **3**          | **1,098.8 μs** | **819.6 μs** | **212.84 μs** |
 **RmsNorm** | **4**          | **2,028.6 μs** | **621.8 μs** | **161.49 μs** |
