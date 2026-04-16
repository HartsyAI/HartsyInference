using BenchmarkDotNet.Running;
using SharpInference.Benchmarks;

BenchmarkSwitcher.FromTypes(
[
    typeof(MatMulBenchmarks),
    typeof(Conv2DBenchmarks),
    typeof(GroupNormBenchmarks),
]).Run(args);
