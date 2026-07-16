using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.TextEncoders;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Self-consistency checks for the Qwen3-VL multimodal encode path used by Boogu-Image editing. For a
/// text-only prompt the 3D M-RoPE position ids collapse to <c>(s, s, s)</c>, so
/// <see cref="Qwen3VlMultimodalEncoder.Encode"/> must reproduce <see cref="LlamaStyleEncoder.Encode"/> exactly —
/// any divergence indicts the embeds/M-RoPE path that only the edit flow exercises (T2I uses the standard encode).</summary>
public sealed class Qwen3VlMultimodalConsistencyTests
{
    private readonly ITestOutputHelper _output;
    public Qwen3VlMultimodalConsistencyTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void MultimodalEncoder_TextOnly_MatchesStandardEncode()
    {
        LlamaStyleEncoderConfig lcfg = new()
        {
            HiddenSize = 24,
            NumLayers = 3,
            NumQueryHeads = 2,
            NumKvHeads = 1,
            HeadDim = 12,
            IntermediateSize = 32,
            VocabSize = 200,
            RmsNormEps = 1e-6f,
            RopeTheta = 5_000_000f,
            MaxPositionEmbeddings = 512,
            QkHeadNorm = true,
            AttentionBias = false,
            HasFinalNorm = true,
        };
        Qwen3VlVisionConfig vcfg = new()
        {
            Depth = 2,
            HiddenSize = 16,
            NumHeads = 2,
            IntermediateSize = 32,
            InChannels = 3,
            PatchSize = 16,
            SpatialMergeSize = 2,
            TemporalPatchSize = 2,
            OutHiddenSize = 24,
            NumPositionEmbeddings = 16,
            DeepstackVisualIndexes = [1],
            NormEps = 1e-6f,
        };

        using CpuBackend backend = new();
        using LlamaStyleEncoder lm = new(lcfg);
        Dictionary<string, Tensor> lw = Qwen3VlVisionTowerTests.BuildLmWeights(lcfg);
        lm.LoadWeights(lw);

        using Qwen3VlVisionEncoder vision = new(vcfg);
        Qwen3VlImageProcessor proc = new(vcfg, maxPixels: 64 * 64);
        Qwen3VlMultimodalEncoder mm = new(lm, vision, proc, vcfg, imageTokenId: 100,
            textHeadDim: lcfg.HeadDim, ropeTheta: lcfg.RopeTheta, mropeSection: [2, 2, 2]);

        int[] tokens = [1, 2, 3, 42, 17, 5, 99, 123];
        using Tensor viaMrope = mm.Encode(backend, tokens, []);
        using Tensor viaStandard = lm.Encode(backend, [tokens]);

        Assert.Equal(viaStandard.Shape, viaMrope.Shape);
        float* a = (float*)viaStandard.DataPointer;
        float* b = (float*)viaMrope.DataPointer;
        long n = viaStandard.ElementCount;
        double maxAbs = 0;
        for (long i = 0; i < n; i++)
        {
            double d = Math.Abs(a[i] - b[i]);
            if (d > maxAbs) maxAbs = d;
        }
        _output.WriteLine($"text-only max abs diff standard vs M-RoPE embeds path: {maxAbs:E3}");
        Assert.True(maxAbs < 1e-5, $"EncodeEmbedsMrope diverges from Encode on a text-only prompt (maxAbs={maxAbs:E3}).");

        foreach (Tensor t in lw.Values) t.Dispose();
    }
}
