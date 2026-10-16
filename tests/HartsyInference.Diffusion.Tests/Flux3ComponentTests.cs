using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>FLUX.3 contracts that fail silently when wrong: rotary position ids, dims read from weights, the shared
/// latent layout, prompt padding and its mask, span modulation, the layout table and structural variant selection.</summary>
public sealed unsafe class Flux3ComponentTests
{
    [Fact]
    public void ImageIds_ScaleRowAndColumn_AtTheGivenIndex()
    {
        using Tensor ids = Flux3PosEmbed.ImageIds(2, 3, 0f, 0.5f);
        float* p = (float*)ids.DataPointer;
        Assert.Equal(new TensorShape(6, 4), ids.Shape);
        // Token (row 1, col 2) is the last one.
        Assert.Equal([0f, 0.5f, 1f, 0f], new[] { p[20], p[21], p[22], p[23] });
    }

    [Fact]
    public void ReferenceIds_SitOnNegativeLAxis_InOrder()
    {
        using Tensor ids = Flux3PosEmbed.ReferenceIds([(1, 2), (2, 1)], 0.5f);
        float* p = (float*)ids.DataPointer;
        Assert.Equal(new TensorShape(4, 4), ids.Shape);
        Assert.Equal(-1f, p[3]);
        Assert.Equal(-1f, p[7]);
        Assert.Equal(-2f, p[11]);
        Assert.Equal([0f, 0.5f, 0f, -2f], new[] { p[12], p[13], p[14], p[15] });
    }

    [Fact]
    public void TextIds_CountAlongLAxisOnly()
    {
        using Tensor ids = Flux3PosEmbed.TextIds(3);
        float* p = (float*)ids.DataPointer;
        Assert.Equal([0f, 0f, 0f, 2f], new[] { p[8], p[9], p[10], p[11] });
    }

    [Fact]
    public void Config_ReadsEveryVaryingDimFromWeights()
    {
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>
        {
            ["content_mode_blocks.image.0.attn_out.weight"] = new Tensor(new TensorShape(256, 1), DType.F32),
            ["content_mode_blocks.image.0.mlp_in.weight"] = new Tensor(new TensorShape(1536, 1), DType.F32),
            ["txt_mode_blocks.0.attn_out.weight"] = new Tensor(new TensorShape(1), DType.F32),
            ["txt_mode_blocks.1.attn_out.weight"] = new Tensor(new TensorShape(1), DType.F32),
            ["single_blocks.0.attn_out.weight"] = new Tensor(new TensorShape(1), DType.F32),
            ["single_blocks.1.attn_out.weight"] = new Tensor(new TensorShape(1), DType.F32),
            ["single_blocks.2.attn_out.weight"] = new Tensor(new TensorShape(1), DType.F32),
            ["ref_time_in.in_layer.weight"] = new Tensor(new TensorShape(1), DType.F32),
        };
        try
        {
            Flux3Config config = Flux3Config.FromWeights(weights);
            Assert.Equal(256, config.HiddenSize);
            Assert.Equal(2, config.NumHeads);
            Assert.Equal(768, config.MlpHiddenSize);
            Assert.Equal(2, config.Depth);
            Assert.Equal(3, config.DepthSingleBlocks);
            Assert.True(config.FlowMap);
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }

    [Fact]
    public void LatentOps_RoundTripExactly()
    {
        using Tensor latent = Fill(new TensorShape(1, 2, 4, 6), i => i * 0.25f - 3f);
        using Tensor mean = Fill(new TensorShape(8), i => 0.1f * i);
        using Tensor var = Fill(new TensorShape(8), i => 1f + i);
        using Tensor tokens = Flux2LatentOps.ToPackedTokens(latent, mean, var, 1e-5f, 2);
        Assert.Equal(new TensorShape(1, 6, 8), tokens.Shape);
        using Tensor back = Flux2LatentOps.FromPackedTokens(tokens, 2, 3, mean, var, 1e-5f, 2, 2);
        Assert.Equal(latent.Shape, back.Shape);
        ReadOnlySpan<float> a = latent.AsReadOnlySpan<float>(), b = back.AsReadOnlySpan<float>();
        for (int i = 0; i < a.Length; i++) Assert.Equal(a[i], b[i], 4);
    }

    [Fact]
    public void Patchify_PutsSubPixelInChannelOrder()
    {
        using Tensor latent = Fill(new TensorShape(1, 1, 2, 2), i => i);
        using Tensor patched = Flux2LatentOps.Patchify(latent, 4, 2);
        Assert.Equal([0f, 1f, 2f, 3f], patched.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void PadToMultiple_PadsWithWeightOne_AndLeavesAlignedSequences()
    {
        WeightedTokenSequence sequence = new WeightedTokenSequence([1, 2, 3], [1.5f, 1f, 0.5f]);
        WeightedTokenSequence padded = sequence.PadToMultiple(4, 9);
        Assert.Equal([1, 2, 3, 9], padded.Tokens);
        Assert.Equal([1.5f, 1f, 0.5f, 1f], padded.Weights);
        Assert.Same(padded, padded.PadToMultiple(4, 9));
        Assert.Equal(160, new WeightedTokenSequence(new int[81], new float[81]).PadToMultiple(80, 0).Tokens.Length);
    }

    [Fact]
    public void PaddedMask_HidesPaddingKeysFromEveryRow()
    {
        using Tensor mask = TextEncoderTensorHelpers.BuildCausalMask(4, int.MaxValue, keyLimit: 2);
        float* p = (float*)mask.DataPointer;
        Assert.Equal(0f, p[0 * 4 + 0]);
        Assert.True(p[1 * 4 + 2] < -1e29f);
        // A padding row still sees the real tokens but not itself.
        Assert.Equal(0f, p[3 * 4 + 1]);
        Assert.True(p[3 * 4 + 3] < -1e29f);
    }

    [Fact]
    public void GatherRopeRows_ParksPaddingAtItsPosition()
    {
        float[] table = [0f, 0f, 10f, 11f, 20f, 21f, 30f, 31f];
        TextEncoderPadding padding = TextEncoderPadding.Transformers(validLength: 2);
        float[] gathered = TextEncoderTensorHelpers.GatherRopeRows(table, 2, 4, padding.PositionOf);
        Assert.Equal([0f, 0f, 10f, 11f, 10f, 11f, 10f, 11f], gathered);
    }

    [Fact]
    public void SpanModulation_MatchesUniform_WhenEverySpanSharesRows()
    {
        IBackend backend = new CpuBackend();
        using Tensor x = Fill(new TensorShape(1, 5, 8), i => MathF.Sin(i));
        using Tensor value = Fill(new TensorShape(1, 5, 8), i => MathF.Cos(i));
        using Tensor shift = Fill(new TensorShape(1, 8), i => 0.1f * i);
        using Tensor scale = Fill(new TensorShape(1, 8), i => -0.05f * i);
        using Tensor gate = Fill(new TensorShape(1, 8), i => 0.3f + 0.01f * i);
        using Flux3Modulation spans = Flux3Modulation.Spans(backend, [(shift, scale, gate), (shift, scale, gate)], [2, 3]);
        using Tensor uniformShift = Copy(shift), uniformScale = Copy(scale), uniformGate = Copy(gate);
        using Flux3Modulation uniform = Flux3Modulation.Uniform(uniformShift, uniformScale, uniformGate);
        using Tensor a = spans.NormModulate(backend, x, 1e-6f);
        using Tensor b = uniform.NormModulate(backend, x, 1e-6f);
        AssertClose(a, b);
        using Tensor c = spans.GatedResidual(backend, x, value);
        using Tensor d = uniform.GatedResidual(backend, x, value);
        AssertClose(c, d);
    }

    [Fact]
    public void LayoutTable_MatchesComfyFlux3LayoutPrompt()
    {
        (string prompt, string? table) = LayoutPromptBuilder.Split("a cat <region:0.1,0.2,0.3,0.4> (red:1.3) \"ball\"");
        Assert.Equal("a cat", prompt);
        Assert.Equal("[{\"id\": \"obj_1\", \"bbox\": [200, 100, 600, 400], \"desc\": \"red \\\"ball\\\"\"}]", table);
        Assert.Equal("plain prompt", LayoutPromptBuilder.Build("plain prompt"));
    }

    [Fact]
    public void NormalizedBox_ClampsToTheGrid()
    {
        Assert.Equal((0, 900, 1000, 1000), LayoutPromptBuilder.NormalizedBox(0.9f, -0.1f, 0.5f, 1.5f));
    }

    [Fact]
    public void Variant_RefTimeIn_ProvesDistilled_AndNothingElseCanClaimIt()
    {
        CheckpointProbe distilled = CheckpointProbe.Empty with { Keys = new HashSet<string>(["ref_time_in.in_layer.weight"]) };
        ResolvedModelVariant structural = ModelVariantResolver.Classify(Flux3Variants.Catalog, distilled, []);
        Assert.True(structural.Is(Flux3Variants.Distilled));
        Assert.Equal(ModelVariantSource.Structure, structural.Source);
        ResolvedModelVariant hinted = ModelVariantResolver.Classify(Flux3Variants.Catalog, CheckpointProbe.Empty, ["distilled"]);
        Assert.True(hinted.Is(Flux3Variants.Base));
        Assert.Equal(Flux3Recipe.DistilledDefaults, new Flux3Recipe().DefaultsFor(structural));
    }

    [Theory]
    // On-grid steps jump to the next entry; the terminal sigma has nowhere further to go.
    [InlineData(1.0f, 0.75f)]
    [InlineData(0.75f, 0.4f)]
    [InlineData(0.0f, 0.0f)]
    // A sub-step evaluation (Heun's corrector at sigma_next, a midpoint) takes the entry nearest it, as ComfyUI does.
    [InlineData(0.6f, 0.4f)]
    [InlineData(0.55f, 0.1f)]
    public void NextScheduleSigma_FollowsComfyFlowMapTargetRule(float sigma, float expected)
    {
        float[] schedule = [1.0f, 0.75f, 0.4f, 0.1f, 0.0f];
        Assert.Equal(expected, Flux3Pipeline.NextScheduleSigma(schedule, sigma));
    }

    [Fact]
    public void NextScheduleSigma_ReadsTheSamplersOwnSchedule()
    {
        // A re-spaced schedule: the target is its next entry, not the family grid's.
        ISampler sampler = SamplerRegistry.Create("euler", [1.0f, 0.9f, 0.3f, 0.0f], seed: 0, options: null);
        Assert.Equal(0.3f, Flux3Pipeline.NextScheduleSigma(sampler.Sigmas, 0.9f));
    }

    [Fact]
    public void AddNoiseAtSigma_MatchesTheIndexedAddNoise()
    {
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(6.93f);
        scheduler.SetTimesteps(4);
        using Tensor sample = Fill(new TensorShape(1, 3, 2), i => i * 0.5f);
        using Tensor noise = Fill(new TensorShape(1, 3, 2), i => 1f - i);
        using Tensor indexed = new Tensor(sample.Shape, DType.F32);
        using Tensor explicitSigma = new Tensor(sample.Shape, DType.F32);
        scheduler.AddNoise(indexed, sample, noise, 2);
        FlowMatchEulerDiscreteScheduler.AddNoiseAtSigma(explicitSigma, sample, noise, scheduler.Sigmas()[2]);
        Assert.Equal(indexed.AsReadOnlySpan<float>().ToArray(), explicitSigma.AsReadOnlySpan<float>().ToArray());
    }

    private static Tensor Fill(TensorShape shape, Func<int, float> value)
    {
        Tensor t = new Tensor(shape, DType.F32);
        Span<float> span = t.AsSpan<float>();
        for (int i = 0; i < span.Length; i++) span[i] = value(i);
        return t;
    }

    private static Tensor Copy(Tensor source) => Fill(source.Shape, i => source.AsReadOnlySpan<float>()[i]);

    private static void AssertClose(Tensor expected, Tensor actual)
    {
        ReadOnlySpan<float> a = expected.AsReadOnlySpan<float>(), b = actual.AsReadOnlySpan<float>();
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++) Assert.True(MathF.Abs(a[i] - b[i]) <= 1e-5f, $"element {i}: {a[i]} vs {b[i]}");
    }
}
