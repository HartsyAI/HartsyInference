using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Expert offload on a real device: a <see cref="CudaExpertCache"/> holds some Q4_K experts, the packed CPU kernels run the
/// rest, and the layer's output is held to the direct path, where every expert runs on the device. The CPU side quantizes its
/// activations to int8, so the bound is the packed kernels' tolerance, not equality.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("CudaSerial")]
public sealed class CudaMoeOffloadTests(ITestOutputHelper output)
{
    private const int Hidden = 256;
    private const int Intermediate = 256;
    private const int Experts = 8;
    private const int TopK = 2;
    private const string Prefix = "blk.0";

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    [Theory]
    [InlineData(1, 3)]    // decode, a mixed split
    [InlineData(12, 3)]   // a batch, a mixed split
    [InlineData(12, 0)]   // a batch, every expert off the device
    public void OffloadMatchesTheDirectPath_WithinThePackedKernelTolerance(int tokens, int residentExperts)
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend cuda = new(0, PtxDir());
        Dictionary<string, Tensor> weights = Q4KWeights(seed: 13);
        MoeConfig moe = new() { NumExperts = Experts, NumExpertsPerTok = TopK, MoeIntermediateSize = Intermediate, Scoring = MoeScoring.Softmax };
        using Tensor x = Random(new Random(4), tokens, Hidden, 1f);

        MoeFeedForward direct = new(moe, Hidden, lowVram: true);
        direct.LoadWeights(weights, Prefix);
        float[] expected = direct.Forward(cuda, x, tokens).AsReadOnlySpan<float>().ToArray();

        long expertBytes = DType.Q4_K.ComputeByteCount((long)Intermediate * Hidden) * 3;
        MoeFeedForward routed = new(moe, Hidden, lowVram: true);
        routed.LoadWeights(weights, Prefix);
        List<Tensor> expertTensors = [.. routed.EnumerateExpertGroups().SelectMany(static g => g)];
        using MoeExpertOffload offload = CudaMoeOffload.Create(cuda, Math.Max(1, residentExperts * expertBytes), Hidden, Intermediate, expertTensors);
        routed.AttachOffload(offload, layer: 0);
        offload.Seed([(0, Experts)], residentExperts);
        float[] actual = routed.Forward(cuda, x, tokens).AsReadOnlySpan<float>().ToArray();

        float maxAbs = 0f, maxRef = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(actual[i] - expected[i]));
            maxRef = MathF.Max(maxRef, MathF.Abs(expected[i]));
        }
        MoeOffloadStats stats = offload.Stats;
        output.WriteLine($"tokens={tokens} resident={residentExperts} relative={maxAbs / maxRef:E3} {stats}");
        Assert.Equal(tokens * TopK, stats.ResidentRows + stats.StreamedRows + stats.HostRows);
        if (residentExperts == 0) Assert.Equal(0, stats.ResidentRows);
        Assert.True(maxAbs / maxRef <= 2.5e-2f, $"relative error {maxAbs / maxRef:E3}");
    }

    private static Dictionary<string, Tensor> Q4KWeights(int seed)
    {
        Random random = new(seed);
        Dictionary<string, Tensor> weights = new() { [$"{Prefix}.mlp.gate.weight"] = Random(random, Experts, Hidden, 0.5f) };
        // Each projection is one stacked tensor viewed per expert, as the GGUF loader produces them.
        Tensor Stack(int rows, int cols) { using Tensor f = Random(random, Experts * rows, cols, 0.1f); return GgufQuantizer.Quantize(f, DType.Q4_K); }
        Tensor gate = Stack(Intermediate, Hidden), up = Stack(Intermediate, Hidden), down = Stack(Hidden, Intermediate);
        for (int e = 0; e < Experts; e++)
        {
            weights[$"{Prefix}.mlp.experts.{e}.gate_proj.weight"] = gate.SliceRows(e * Intermediate, Intermediate);
            weights[$"{Prefix}.mlp.experts.{e}.up_proj.weight"] = up.SliceRows(e * Intermediate, Intermediate);
            weights[$"{Prefix}.mlp.experts.{e}.down_proj.weight"] = down.SliceRows(e * Hidden, Hidden);
        }
        return weights;
    }

    private static Tensor Random(Random random, int rows, int cols, float scale)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        Span<float> data = t.AsSpan<float>();
        for (int i = 0; i < data.Length; i++) data[i] = (float)(random.NextDouble() * 2 - 1) * scale;
        return t;
    }
}
