using BenchmarkDotNet.Running;
using HartsyInference.Benchmarks;

BenchmarkSwitcher.FromTypes(
[
    typeof(MatMulBenchmarks),
    typeof(Conv2DBenchmarks),
    typeof(GroupNormBenchmarks),
]).Run(args);
