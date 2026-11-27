using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A per-row INT8 LongCat release against the BF16 release it came from (<see cref="LongCatInt8ReleaseCheck"/>): the check on tiny
/// releases, with weights it must fail on, and the check on the released Avatar files. The released-file case runs on the CPU and only when
/// HARTSY_LONGCAT_INT8_INDEX and HARTSY_LONGCAT_BF16_INDEX name the two index files, each beside its shards.</summary>
public sealed class LongCatInt8ReleaseTests(ITestOutputHelper output) : IDisposable
{
    /// <summary>The quantized linear types of the avatar DiT: thirteen per block, the two timestep and two caption linears, the final adaLN
    /// projection and four audio projections.</summary>
    private const int AvatarLinearTypes = 22;

    private readonly string _root = Path.Combine(Path.GetTempPath(), "longcat-int8-release-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    [Trait("Category", "Integration")]
    [RealWeightsFact("HARTSY_LONGCAT_INT8_INDEX", "HARTSY_LONGCAT_BF16_INDEX")]
    public void TheReleasedInt8WeightsAreTheReleasedBf16WeightsWithinHalfAStep()
    {
        List<LongCatInt8ReleaseCheck.LinearReport> reports = LongCatInt8ReleaseCheck.Run(
            Environment.GetEnvironmentVariable("HARTSY_LONGCAT_INT8_INDEX")!,
            Environment.GetEnvironmentVariable("HARTSY_LONGCAT_BF16_INDEX")!);

        foreach (LongCatInt8ReleaseCheck.LinearReport report in reports)
            output.WriteLine(report.ToString());
        Assert.Equal(AvatarLinearTypes, reports.Count);
        Assert.All(reports, static r => Assert.Equal((0L, 0L), (r.ReleaseViolations, r.EngineViolations)));
    }

    [Trait("Category", "SyntheticSmoke")]
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ATinyReleasePasses(bool quantizedFromBf16)
    {
        List<LongCatInt8ReleaseCheck.LinearReport> reports = Tiny(quantizedFromBf16, static (_, _) => { });

        // t_embedder two, y_embedder two, the final adaLN projection and nine linears per block, whatever the depth.
        Assert.Equal(2 + 2 + 1 + 9, reports.Count);
        Assert.All(reports, static r => Assert.Equal((0L, 0L), (r.ReleaseViolations, r.EngineViolations)));
        Assert.Contains(reports, static r => r.Key == "blocks.0.attn.qkv");
        // Half a step when quantized from the stored values; the BF16 rounding of the stored weights adds to it otherwise.
        Assert.All(reports, r => Assert.InRange(r.MaxStepError, 0.3, quantizedFromBf16 ? 0.5001 : 1.0));
    }

    [Trait("Category", "SyntheticSmoke")]
    [Fact]
    public void AScaleTwoPercentOffIsCaught()
    {
        List<LongCatInt8ReleaseCheck.LinearReport> reports = Tiny(quantizedFromBf16: false, static (int8, _) =>
            int8["blocks.0.ffn.w2.weight_scale"].AsSpan<float>()[3] *= 1.02f);

        LongCatInt8ReleaseCheck.LinearReport failed = Assert.Single(reports, static r => r.ReleaseViolations > 0);
        Assert.Equal("blocks.0.ffn.w2", failed.Key);
    }

    [Trait("Category", "SyntheticSmoke")]
    [Fact]
    public void OneLevelThreeStepsOffIsCaught()
    {
        List<LongCatInt8ReleaseCheck.LinearReport> reports = Tiny(quantizedFromBf16: false, static (int8, _) =>
        {
            Span<byte> levels = int8["blocks.0.attn.proj.weight_int8"].AsSpan<byte>();
            sbyte level = unchecked((sbyte)levels[17]);
            levels[17] = unchecked((byte)(sbyte)(level > 0 ? level - 3 : level + 3));
        });

        LongCatInt8ReleaseCheck.LinearReport failed = Assert.Single(reports, static r => r.ReleaseViolations > 0);
        Assert.Equal(("blocks.0.attn.proj", 1L), (failed.Key, failed.ReleaseViolations));
    }

    [Trait("Category", "SyntheticSmoke")]
    [Fact]
    public void AMissingDenseWeightIsRefused()
    {
        InvalidDataException refused = Assert.Throws<InvalidDataException>(() =>
            Tiny(quantizedFromBf16: false, static (_, dense) => dense.Remove("blocks.0.ffn.w1.weight")));

        Assert.Contains("blocks.0.ffn.w1.weight", refused.Message, StringComparison.Ordinal);
    }

    private static Tensor RoundedBf16(Tensor source)
    {
        Tensor rounded = new(source.Shape, DType.BF16);
        ReadOnlySpan<float> from = source.AsReadOnlySpan<float>();
        Span<ushort> to = rounded.AsSpan<ushort>();
        for (int i = 0; i < from.Length; i++)
            to[i] = BFloat16Bits.FromFloat(from[i]);
        return rounded;
    }

    /// <summary>Writes a tiny BF16 release and its INT8 form, lets <paramref name="damage"/> change either, and runs the check.</summary>
    private List<LongCatInt8ReleaseCheck.LinearReport> Tiny(bool quantizedFromBf16,
        Action<Dictionary<string, Tensor>, Dictionary<string, Tensor>> damage)
    {
        string int8Folder = Directory.CreateDirectory(Path.Combine(_root, "int8")).FullName;
        string denseFolder = Directory.CreateDirectory(Path.Combine(_root, "dense")).FullName;
        Dictionary<string, Tensor> full = LongCatTestData.Weights(LongCatTestData.Tiny, seed: 33);
        // Rounded to nearest, as the released BF16 files are.
        Dictionary<string, Tensor> dense = full.ToDictionary(static p => p.Key, static p => RoundedBf16(p.Value), StringComparer.Ordinal);
        Dictionary<string, Tensor> source = quantizedFromBf16
            ? dense.ToDictionary(static p => p.Key, static p => p.Value.CastTo(DType.F32), StringComparer.Ordinal)
            : full;
        Dictionary<string, Tensor> int8 = LongCatTestData.QuantizePerRow(source);
        List<Tensor> owned = [.. full.Values.Concat(dense.Values).Concat(source.Values).Concat(int8.Values).Distinct()];
        try
        {
            damage(int8, dense);
            LongCatTestData.WriteCheckpoint(int8Folder, int8, indexFileName: LongCatCheckpoint.Int8IndexFileName);
            LongCatTestData.WriteCheckpoint(denseFolder, dense);
            return LongCatInt8ReleaseCheck.Run(Path.Combine(int8Folder, LongCatCheckpoint.Int8IndexFileName),
                Path.Combine(denseFolder, LongCatCheckpoint.IndexFileName));
        }
        finally
        {
            foreach (Tensor tensor in owned)
                tensor.Dispose();
        }
    }
}
