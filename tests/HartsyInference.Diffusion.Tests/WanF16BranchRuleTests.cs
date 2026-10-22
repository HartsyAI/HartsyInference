using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common.Attention;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Which Wan forwards take F16 branches: TI2V-5B and a single-expert 14B T2V, one timestep group, under
/// <c>numerics.wanF16</c> (on by default, off in the <c>reference</c> profile) and <c>numerics.ditF16</c>, on a backend with F16
/// activations. An A14B expert pair, image-to-video and the 1.3B stay F32, the weight detector grants the same geometries as
/// the presets, and with the switch on a forward with per-frame timesteps, an image context, context parallelism or the step
/// cache stays F32.</summary>
public sealed class WanF16BranchRuleTests
{
    [Fact]
    public void TheSwitchIsOnByDefaultAndTheReferenceProfileKeepsF32()
    {
        using WanVideoTransformer transformer = new(WanVideoConfig.Ti2V5B);
        AttentionTraceBackend backend = new() { SupportsF16Activations = true };

        Assert.True(EngineKnobs.WanF16.Default);
        using (KnobProfiles.Reference.Push())
            Assert.Equal(DType.F32, transformer.BranchActivation(backend, singleGroup: true));
    }

    [Theory]
    [InlineData("plain", true)]
    [InlineData("per-frame timesteps", false)]
    [InlineData("image context", false)]
    [InlineData("context parallelism", false)]
    [InlineData("step cache", false)]
    [InlineData("A14B pair", false)]
    public void WithTheSwitchOnOnlyAPlainSingleGroupForwardTakesF16Branches(string path, bool f16)
    {
        WanVideoConfig cfg = path == "A14B pair" ? WanVideoConfig.T2V_14B with { BoundaryRatio = 0.875f } : WanVideoConfig.Ti2V5B;
        using WanVideoTransformer transformer = new(cfg);
        AttentionTraceBackend backend = new() { SupportsF16Activations = true };
        using IDisposable knobs = KnobProfile.Create("wan-f16-forward").With(EngineKnobs.WanF16, true).With(EngineKnobs.DitF16, true).Push();

        DType act = transformer.ForwardActivation(backend, timestepGroups: path == "per-frame timesteps" ? 3 : 1,
            imageContext: path == "image context", contextParallel: path == "context parallelism", stepCache: path == "step cache");

        Assert.Equal(f16 ? DType.F16 : DType.F32, act);
    }

    [Theory]
    [InlineData("ti2v-5b", true)]
    [InlineData("t2v-14b", true)]
    [InlineData("t2v-14b-pair", false)]
    [InlineData("t2v-a14b", false)]
    [InlineData("i2v-a14b", false)]
    [InlineData("i2v-14b-clip", false)]
    [InlineData("t2v-1.3b", false)]
    public void OnlyTi2v5bAndASingle14bT2vTakeF16BranchesAndOnlyUnderBothSwitches(string name, bool eligible)
    {
        WanVideoConfig cfg = name switch
        {
            "ti2v-5b" => WanVideoConfig.Ti2V5B,
            "t2v-14b" => WanVideoConfig.T2V_14B,
            "t2v-14b-pair" => WanVideoConfig.T2V_14B with { BoundaryRatio = 0.875f },
            "t2v-a14b" => WanVideoConfig.T2V_A14B,
            "i2v-a14b" => WanVideoConfig.I2V_A14B,
            "i2v-14b-clip" => WanVideoConfig.I2V_14B_480p,
            _ => WanVideoConfig.T2V_1_3B,
        };
        using WanVideoTransformer transformer = new(cfg);
        AttentionTraceBackend f16 = new() { SupportsF16Activations = true }, f32Only = new();

        DType Branch(bool wanF16, bool ditF16, IBackend backend, bool singleGroup = true)
        {
            using IDisposable knobs = KnobProfile.Create("wan-f16-rule").With(EngineKnobs.WanF16, wanF16)
                .With(EngineKnobs.DitF16, ditF16).Push();
            return transformer.BranchActivation(backend, singleGroup);
        }

        Assert.Equal(eligible ? DType.F16 : DType.F32, Branch(true, true, f16));
        Assert.Equal(DType.F32, Branch(false, true, f16));
        Assert.Equal(DType.F32, Branch(true, false, f16));
        Assert.Equal(DType.F32, Branch(true, true, f32Only));
        Assert.Equal(DType.F32, Branch(true, true, f16, singleGroup: false));
    }

    [Theory]
    [InlineData(3072, 48, "", true)]
    [InlineData(5120, 16, "", true)]
    [InlineData(5120, 36, "", false)]
    [InlineData(5120, 36, "clip", false)]
    [InlineData(5120, 16, "vace", false)]
    [InlineData(1536, 16, "", false)]
    public void TheDetectorGrantsTheSameGeometries(int inner, int inChannels, string extra, bool eligible)
    {
        static Tensor T(params long[] shape) => new(new TensorShape(shape), DType.F16);
        int outChannels = inChannels == 48 ? 48 : 16;
        Dictionary<string, Tensor> w = new()
        {
            ["patch_embedding.weight"] = T(inner, inChannels, 1, 2, 2),
            ["proj_out.weight"] = T(outChannels * 4, inner),
            ["blocks.0.ffn.net.0.proj.weight"] = T(8, inner),
        };
        if (extra == "clip") w["condition_embedder.image_embedder.norm1.weight"] = T(1280);
        if (extra == "vace")
        {
            w["vace_patch_embedding.weight"] = T(inner, 96, 1, 2, 2);
            w["vace_blocks.0.proj_out.weight"] = T(8, 8);
        }
        try
        {
            Assert.Equal(eligible, WanConfigDetector.Detect(w).F16Branches);
        }
        finally
        {
            foreach (Tensor t in w.Values) t.Dispose();
        }
    }
}
