using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Shared;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Block 0 of a released BF16 LongCat-Video-Avatar DiT, self-attention, text cross-attention, audio step and FFN, on Vulkan against the CPU
/// backend: the released weights, seeded inputs on a 3 x 8 x 8 token grid with one conditioning frame. Two lanes: GEMMs in F32
/// (<see cref="VulkanBackend.EnableF16Gemm"/> off) and the backend's default F16 GEMMs. Runs only when HARTSY_LONGCAT_BLOCK_INDEX names the
/// release's dense index, beside at least block 0's shards and <c>config.json</c>, and a Vulkan device opens.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatRealBlockVulkanTests(ITestOutputHelper output)
{
    private const string EnvName = "HARTSY_LONGCAT_BLOCK_INDEX";
    private const int Frames = 3, GridH = 8, GridW = 8, CondFrames = 1, TextRows = 32;

    /// <summary>The relative RMS of the output against the CPU's, GEMMs in F32. Set before the first run on real weights; measured 1.846e-6 on an
    /// RTX 3060 (2026-11-27).</summary>
    private const double ExactBound = 1e-4;

    /// <summary>The same with the default F16 GEMMs. Set before the first run on real weights; measured 4.259e-4 on the same card.</summary>
    private const double DefaultBound = 1e-2;

    [RealBlockFact]
    public void BlockZeroWithF32GemmsMatchesTheCpu() => Compare(f16Gemm: false, ExactBound);

    [RealBlockFact]
    public void BlockZeroWithDefaultGemmsStaysWithinTheF16Bound() => Compare(f16Gemm: true, DefaultBound);

    private void Compare(bool f16Gemm, double bound)
    {
        string index = Environment.GetEnvironmentVariable(EnvName)!;
        string directory = Path.GetDirectoryName(Path.GetFullPath(index))!;   // a full file path always has a directory
        LongCatConfig config = LongCatConfig.FromJson(File.ReadAllText(Path.Combine(directory, "config.json")));
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(index));
        Dictionary<string, string> block = document.RootElement.GetProperty("weight_map").EnumerateObject()
            .Where(static p => p.Name.StartsWith("blocks.0.", StringComparison.Ordinal))
            .ToDictionary(static p => p.Name, static p => p.Value.GetString()!, StringComparer.Ordinal);
        using ShardedSafeTensorSet shards = ShardedSafeTensorSet.OpenFiles(
            [.. block.Values.Distinct(StringComparer.Ordinal).Select(file => Path.Combine(directory, file))]);
        Dictionary<string, Tensor> weights = block.Keys.ToDictionary(static key => key, shards.GetTensor, StringComparer.Ordinal);
        Assert.Equal(DType.BF16, weights["blocks.0.attn.qkv.weight"].DType);
        Assert.Contains("blocks.0.audio_cross_attn.kv_linear.weight", weights.Keys);

        float[] expected, input;
        using (CpuBackend cpu = new())
            expected = Forward(cpu, config, weights, out input);
        using VulkanBackend vulkan = GpuDevices.OpenVulkan();
        vulkan.EnableF16Gemm = f16Gemm;
        float[] actual = Forward(vulkan, config, weights, out _);

        Assert.All(expected, static v => Assert.True(float.IsFinite(v)));
        int bad = actual.Count(static v => !float.IsFinite(v));
        double relRms = LongCatTestData.RelRms(actual, expected);
        double rms = Math.Sqrt(expected.Sum(static v => (double)v * v) / expected.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
            worst = Math.Max(worst, Math.Abs((double)actual[i] - expected[i]));
        double moved = LongCatTestData.RelRms(expected, input);
        int at = 0;
        double peak = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            peak = Math.Max(peak, Math.Abs(expected[i]));
            if (Math.Abs((double)actual[i] - expected[i]) >= worst)
                at = i;
        }
        // The worst row's own relative RMS, which the whole output's would average away.
        double worstRow = 0;
        for (int start = 0; start < expected.Length; start += config.HiddenSize)
        {
            worstRow = Math.Max(worstRow, LongCatTestData.RelRms(actual.AsSpan(start, config.HiddenSize),
                expected.AsSpan(start, config.HiddenSize)));
        }
        output.WriteLine(FormattableString.Invariant(
            $"CPU output: largest magnitude {peak:E3}, value at the worst difference {expected[at]:E3} (Vulkan {actual[at]:E3}), ")
            + FormattableString.Invariant($"worst row relRMS {worstRow:E3}"));
        output.WriteLine(FormattableString.Invariant(
            $"{vulkan.Capabilities.Name}, F16 GEMMs {f16Gemm}: relRMS {relRms:E3} (bound {bound:E0}), worst absolute difference {worst:E3} ")
            + FormattableString.Invariant($"at an output RMS of {rms:E3}, non-finite {bad}, the block moved its input by relRMS {moved:E3}"));
        Assert.Equal(0, bad);
        Assert.True(moved > 1e-3, $"the block left the stream where it was (relRMS {moved})");
        Assert.True(relRms <= bound, $"drifted from the CPU by relRMS {relRms:E3}, above {bound:E0}");
    }

    private static unsafe float[] Forward(IBackend backend, LongCatConfig config, Dictionary<string, Tensor> weights, out float[] input)
    {
        int hw = GridH * GridW, tokens = Frames * hw;
        List<Tensor> owned = [];
        LongCatBlock block = new(config);
        try
        {
            block.LoadWeights(weights, "blocks.0", owned);
            using Tensor hidden = LongCatTestData.Random(new TensorShape(tokens, config.HiddenSize), 21);
            using Tensor text = LongCatTestData.Random(new TensorShape(TextRows, config.HiddenSize), 22);
            using Tensor timeEmbed = LongCatTestData.Random(new TensorShape(Frames, config.AdalnTembedDim), 23);
            using Tensor audio = LongCatTestData.Random(new TensorShape(Frames * config.AudioContextTokens, config.AudioOutputDim), 24);
            input = hidden.AsReadOnlySpan<float>().ToArray();
            WanRope rope = new(config.HeadDim, LongCatConfig.RopeTheta, 0);
            (Tensor cos, Tensor sin) = rope.BuildCosSin(Frames, GridH, GridW);
            using (cos)
            using (sin)
            using (Tensor index = new(new TensorShape(tokens), DType.I32))
            {
                for (int i = 0; i < tokens; i++)
                    ((int*)index.DataPointer)[i] = i / hw;
                // The conditioning rows attend to themselves, the noisy rows to every row.
                int cond = CondFrames * hw;
                AttentionSegment[] segments = [new(0, cond, 0, cond), new(cond, tokens - cond, 0, tokens)];
                using Tensor result = block.Forward(backend, hidden, text, timeEmbed,
                    new LongCatBlockInputs(Frames, hw, CondFrames, cos, sin, segments, index), audio: audio);
                return result.AsReadOnlySpan<float>().ToArray();
            }
        }
        finally
        {
            backend.FreeWeights(block.EnumerateWeights());
            block.Dispose();
            foreach (Tensor tensor in owned)
                tensor.Dispose();
        }
    }

    /// <summary>Runs only when the released index is named and a Vulkan device opens. Otherwise the test is reported as skipped, never
    /// passed.</summary>
    private sealed class RealBlockFactAttribute : FactAttribute
    {
        public RealBlockFactAttribute()
        {
            string? index = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(index) || !File.Exists(index))
                Skip = $"SKIPPED: {EnvName} is unset or not an existing file.";
            else if (GpuDevices.VulkanFailure is { } failure)
                Skip = $"SKIPPED: no usable Vulkan device ({failure}).";
        }
    }
}
