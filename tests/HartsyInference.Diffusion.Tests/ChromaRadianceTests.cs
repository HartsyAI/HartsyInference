using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.Gguf;
using HartsyInference.ModelHandler.Gguf.KeyMappers;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config CPU tests for the Chroma Radiance pixel-space components: conv patchifier shapes, NeRF head
/// round trip (both final-layer variants), the cosine positional basis, x0→velocity conversion, and the
/// detection/converter/key-mapper plumbing. Numerics vs the reference checkpoint are validation-pending.</summary>
public unsafe class ChromaRadianceTests
{
    private const int TokenDim = 16;
    private const int NerfHidden = 8;
    private const int MaxFreqs = 2;
    private const int Depth = 2;
    private const int MlpRatio = 4;
    private const int Patch = 4;

    [Fact]
    public void NerfHead_Forward_RoundTripsImageShape_LinearVariant()
    {
        RunNerfHeadRoundTrip(convFinalLayer: false);
    }

    [Fact]
    public void NerfHead_Forward_RoundTripsImageShape_ConvVariant()
    {
        RunNerfHeadRoundTrip(convFinalLayer: true);
    }

    private static void RunNerfHeadRoundTrip(bool convFinalLayer)
    {
        CpuBackend backend = new();
        Dictionary<string, Tensor> weights = ChromaRadianceSyntheticWeights.BuildNerfHead(
            TokenDim, NerfHidden, MaxFreqs, Depth, MlpRatio, convFinalLayer);

        using ChromaRadianceNerfHead head = new(Patch, NerfHidden, MaxFreqs, Depth, MlpRatio);
        head.LoadWeights(weights);
        Assert.Equal(convFinalLayer, head.UsesConvFinalLayer);

        // 8x8 image with 4-px patches → 2x2 = 4 patches.
        Tensor noisy = Rand4d(1, 3, 8, 8, seed: 21);
        Tensor tokens = Rand3d(1, 4, TokenDim, seed: 22);

        Tensor x0 = head.Forward(backend, noisy, tokens);
        Assert.Equal(1, (int)x0.Shape[0]);
        Assert.Equal(3, (int)x0.Shape[1]);
        Assert.Equal(8, (int)x0.Shape[2]);
        Assert.Equal(8, (int)x0.Shape[3]);
        float* p = (float*)x0.DataPointer;
        for (long i = 0; i < x0.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]));

        x0.Dispose();
        noisy.Dispose();
        tokens.Dispose();
    }

    [Fact]
    public void NerfHead_TokensInfluenceOutput()
    {
        // The hypernetwork must actually route the transformer tokens into the per-patch MLPs:
        // perturbing one patch's token must change that patch's pixels.
        CpuBackend backend = new();
        Dictionary<string, Tensor> weights = ChromaRadianceSyntheticWeights.BuildNerfHead(
            TokenDim, NerfHidden, MaxFreqs, Depth, MlpRatio, convFinalLayer: false);
        using ChromaRadianceNerfHead head = new(Patch, NerfHidden, MaxFreqs, Depth, MlpRatio);
        head.LoadWeights(weights);

        Tensor noisy = Rand4d(1, 3, 8, 8, seed: 31);
        Tensor tokens = Rand3d(1, 4, TokenDim, seed: 32);
        Tensor x0A = head.Forward(backend, noisy, tokens);

        float* tp = (float*)tokens.DataPointer;
        for (int d = 0; d < TokenDim; d++) tp[d] += 1.0f;
        Tensor x0B = head.Forward(backend, noisy, tokens);

        float* a = (float*)x0A.DataPointer;
        float* b = (float*)x0B.DataPointer;
        bool changed = false;
        for (long i = 0; i < x0A.Shape.ElementCount && !changed; i++)
            changed = a[i] != b[i];
        Assert.True(changed, "perturbing a transformer token did not affect the NeRF head output");

        x0A.Dispose();
        x0B.Dispose();
        noisy.Dispose();
        tokens.Dispose();
    }

    [Fact]
    public void Patchifier_Forward_ProducesTokens()
    {
        CpuBackend backend = new();
        using ChromaRadianceImagePatchifier patchifier = new();
        patchifier.LoadWeights(ChromaRadianceSyntheticWeights.BuildPatchifier(hidden: TokenDim, patchSize: Patch));
        Assert.Equal(Patch, patchifier.PatchSize);
        Assert.Equal(TokenDim, patchifier.HiddenSize);

        Tensor rgb = Rand4d(1, 3, 8, 12, seed: 41);
        Tensor tokens = patchifier.Forward(backend, rgb);
        Assert.Equal(1, (int)tokens.Shape[0]);
        Assert.Equal(2 * 3, (int)tokens.Shape[1]);
        Assert.Equal(TokenDim, (int)tokens.Shape[2]);

        tokens.Dispose();
        rgb.Dispose();
    }

    [Fact]
    public void PositionalFeatures_MatchComfyDctBasis()
    {
        // feature[u·F+v](y, x) = cos(π·u·xPos)·cos(π·v·yPos) / (1 + u·v), positions linspace(0, 1, P).
        float[] feat = ChromaRadianceNerfHead.BuildPositionalFeatures(patchSize: 4, maxFreqs: 2);
        Assert.Equal(4 * 4 * 4, feat.Length);

        // Pixel (0,0): all cosines are 1 → features = 1/(1+u·v) = [1, 1, 1, 0.5].
        Assert.Equal(1.0f, feat[0], 5);
        Assert.Equal(1.0f, feat[1], 5);
        Assert.Equal(1.0f, feat[2], 5);
        Assert.Equal(0.5f, feat[3], 5);

        // Pixel (y=0, x=3): xPos=1 → feature[u=1,v=0] = cos(π)·cos(0)/1 = -1; feature[u=0,v=1] = 1.
        int pixBase = (0 * 4 + 3) * 4;
        Assert.Equal(1.0f, feat[pixBase + 1], 5);   // u=0, v=1 → cos(0)·cos(0) = 1 (yPos=0)
        Assert.Equal(-1.0f, feat[pixBase + 2], 5);  // u=1, v=0 → cos(π·1·1) = -1

        // Pixel (y=3, x=0): yPos=1 → feature[u=0,v=1] = cos(π) = -1; feature[u=1,v=0] = 1.
        pixBase = (3 * 4 + 0) * 4;
        Assert.Equal(-1.0f, feat[pixBase + 1], 5);
        Assert.Equal(1.0f, feat[pixBase + 2], 5);
    }

    [Fact]
    public void X0Prediction_ToVelocity_MatchesFormula()
    {
        Tensor x0 = Filled(0.5f);
        Tensor xt = Filled(1.0f);

        Tensor v = X0Prediction.ToVelocity(x0, xt, t: 0.5f);
        Assert.Equal(1.0f, ((float*)v.DataPointer)[0], 5);
        v.Dispose();

        // t = 0 → eps floor keeps the result finite.
        Tensor v0 = X0Prediction.ToVelocity(x0, xt, t: 0.0f);
        Assert.True(float.IsFinite(((float*)v0.DataPointer)[0]));
        v0.Dispose();

        x0.Dispose();
        xt.Dispose();
    }

    [Fact]
    public void Config_DetectsRadianceAndInfersDims()
    {
        Dictionary<string, Tensor> weights = ChromaRadianceSyntheticWeights.BuildNerfHead(
            TokenDim, NerfHidden, MaxFreqs, Depth, MlpRatio, convFinalLayer: false);
        foreach (KeyValuePair<string, Tensor> kvp in ChromaRadianceSyntheticWeights.BuildPatchifier(TokenDim, Patch))
            weights[kvp.Key] = kvp.Value;

        Assert.True(ChromaRadianceConfig.IsRadiance(weights));

        ChromaRadianceConfig config = ChromaRadianceConfig.FromWeights(weights);
        Assert.Equal(Patch, config.PatchSize);
        Assert.Equal(NerfHidden, config.NerfHidden);
        Assert.Equal(MaxFreqs, config.MaxFreqs);
        Assert.Equal(Depth, config.NerfDepth);

        Dictionary<string, Tensor> classic = new() { ["x_embedder.weight"] = Filled(0f) };
        Assert.False(ChromaRadianceConfig.IsRadiance(classic));
    }

    [Fact]
    public void Converter_StripsOrigModPrefix_AndPassesRadianceKeysThrough()
    {
        Dictionary<string, Tensor> raw = new()
        {
            ["_orig_mod.img_in_patch.weight"] = Filled(1f),
            ["_orig_mod.img_in_patch.bias"] = Filled(1f),
            ["_orig_mod.nerf_blocks.0.norm.scale"] = Filled(1f),
            ["_orig_mod.nerf_image_embedder.embedder.0.weight"] = Filled(1f),
            ["_orig_mod.nerf_final_layer_conv.norm.scale"] = Filled(1f),
            ["_orig_mod.distilled_guidance_layer.in_proj.weight"] = Filled(1f),
            ["_orig_mod.txt_in.weight"] = Filled(1f),
        };

        ChromaCheckpointConverter.ConvertedWeights converted = ChromaCheckpointConverter.Convert(raw);
        Assert.True(converted.Transformer.ContainsKey("img_in_patch.weight"));
        Assert.True(converted.Transformer.ContainsKey("img_in_patch.bias"));
        Assert.True(converted.Transformer.ContainsKey("nerf_blocks.0.norm.scale"));
        Assert.True(converted.Transformer.ContainsKey("nerf_image_embedder.embedder.0.weight"));
        Assert.True(converted.Transformer.ContainsKey("nerf_final_layer_conv.norm.scale"));
        Assert.True(converted.Transformer.ContainsKey("distilled_guidance_layer.in_proj.weight"));
        Assert.True(converted.Transformer.ContainsKey("context_embedder.weight"));
        Assert.True(ChromaCheckpointConverter.ContainsRadianceKeys(converted.Transformer));
    }

    [Fact]
    public void KeyMapper_DetectsRadianceBeforeParentFamilies()
    {
        string[] radianceNames =
        [
            "distilled_guidance_layer.norms.0.scale",
            "double_blocks.0.img_attn.qkv.weight",
            "single_blocks.0.linear1.weight",
            "nerf_blocks.0.norm.scale",
            "img_in_patch.weight",
        ];
        IGgufKeyMapper mapper = GgufKeyMapperRegistry.DetectByKeys(radianceNames);
        Assert.Equal("chroma-radiance", mapper.Architecture);

        // Classic Chroma (no NeRF head) must NOT resolve as Radiance.
        string[] classicNames =
        [
            "distilled_guidance_layer.norms.0.scale",
            "double_blocks.0.img_attn.qkv.weight",
            "single_blocks.0.linear1.weight",
            "img_in.weight",
        ];
        IGgufKeyMapper classicMapper = GgufKeyMapperRegistry.DetectByKeys(classicNames);
        Assert.NotEqual("chroma-radiance", classicMapper.Architecture);
    }

    [Fact]
    public void LatentPreview_PixelSpace_IsDirectRgb()
    {
        Assert.True(LatentPreview.IsSupported(LatentArchitecture.ChromaRadiance));
        Assert.True(LatentPreview.IsPixelSpace(LatentArchitecture.ZetaChroma));
        Assert.False(LatentPreview.IsPixelSpace(LatentArchitecture.Chroma));

        Tensor pixels = new Tensor(new TensorShape(1, 3, 2, 2), DType.F32);
        float* p = (float*)pixels.DataPointer;
        for (int i = 0; i < 12; i++) p[i] = 0f;
        p[0] = 1.0f;   // R of pixel 0 → 255
        p[4] = -1.0f;  // G of pixel 0 → 0

        byte[]? rgb = LatentPreview.DecodeLatent2Rgb(pixels, LatentArchitecture.ChromaRadiance, out int w, out int h);
        Assert.NotNull(rgb);
        Assert.Equal(2, w);
        Assert.Equal(2, h);
        Assert.Equal(255, rgb![0]);
        Assert.Equal(0, rgb[1]);
        Assert.Equal(128, rgb[2]);

        pixels.Dispose();
    }

    private static Tensor Rand4d(int b, int c, int h, int w, int seed)
    {
        Tensor t = new Tensor(new TensorShape(b, c, h, w), DType.F32);
        FillRandom(t, seed);
        return t;
    }

    private static Tensor Rand3d(int b, int s, int d, int seed)
    {
        Tensor t = new Tensor(new TensorShape(b, s, d), DType.F32);
        FillRandom(t, seed);
        return t;
    }

    private static Tensor Filled(float value)
    {
        Tensor t = new Tensor(new TensorShape(1, 4), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < 4; i++) p[i] = value;
        return t;
    }

    private static void FillRandom(Tensor t, int seed)
    {
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
    }
}
