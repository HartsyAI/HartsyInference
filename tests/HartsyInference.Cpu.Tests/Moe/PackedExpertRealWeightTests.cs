using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu.Moe;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.MoePack;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Cpu.Tests.Moe.MoeTestData;

namespace HartsyInference.Cpu.Tests.Moe;

/// <summary>
/// Real-weight check of the packed CPU expert path: a Q8_0 expert from a real checkpoint's pack, run through
/// <see cref="PackedExpertHostRunner"/>, against the F32 reference on the same dequantized weights. Set
/// <c>HARTSY_GRANITE_PACK</c> to a completed pack directory and <c>HARTSY_GRANITE_GGUF</c> to its source GGUF.
/// </summary>
[Trait("Category", "RealWeights")]
public sealed class PackedExpertRealWeightTests
{
    private const string PackVar = "HARTSY_GRANITE_PACK";
    private const string GgufVar = "HARTSY_GRANITE_GGUF";
    private readonly ITestOutputHelper _output;

    public PackedExpertRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void RealExperts_PackedKernelMatchesTheF32ReferenceOnTheSameWeights()
    {
        string? packDir = Environment.GetEnvironmentVariable(PackVar);
        string? gguf = Environment.GetEnvironmentVariable(GgufVar);
        if (packDir is null || gguf is null || !RealWeightGate.Require(_output.WriteLine, packDir, gguf)) return;

        using ExpertPackReader reader = ExpertPackReader.Open(packDir, verifyChecksums: true);
        PackedExpertHostRunner runner = new(DType.Q8_0, reader.Hidden, reader.Intermediate, reader.Resolve);
        ExpertKey[] sample = [.. reader.Keys.OrderBy(static k => k.Layer).ThenBy(static k => k.Expert)
            .Where(static (_, i) => i % 97 == 0).Take(4)];
        Assert.NotEmpty(sample);

        foreach (ExpertKey key in sample)
        {
            ExpertWeights weights = reader.Resolve(key);
            F32ExpertWeights reference = new(reader.Hidden, reader.Intermediate,
                Dequantize(weights.W1.Weight), Dequantize(weights.W3.Weight), Dequantize(weights.W2.Weight));
            float[] x = Random(4 * reader.Hidden, seed: key.Layer * 1000 + key.Expert, scale: 1f);
            float[] expected = new float[x.Length];
            ExpertProgramReference.Apply(ExpertProgram.Swiglu, reference, x, 4, expected);
            float[] actual = new float[x.Length];
            runner.Run(ExpertProgram.Swiglu, key, x, 4, actual);

            float maxAbs = 0f;
            float maxRef = 0f;
            for (int i = 0; i < expected.Length; i++)
            {
                maxAbs = MathF.Max(maxAbs, MathF.Abs(actual[i] - expected[i]));
                maxRef = MathF.Max(maxRef, MathF.Abs(expected[i]));
            }
            float relative = maxAbs / maxRef;
            _output.WriteLine($"{key}: relative error {relative:E4}");
            Assert.True(relative <= 2e-2f, $"{key}: relative error {relative:E4} exceeds 2e-2.");
        }
    }

    private static float[] Dequantize(Tensor packed)
    {
        using Tensor f32 = GgufDequantizer.Dequantize(packed, DType.F32);
        return ReadF32(f32);
    }
}
