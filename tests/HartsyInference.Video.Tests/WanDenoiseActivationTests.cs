using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common.Attention;
using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Video.Tests;

/// <summary>The dtype the Wan T2V/TI2V loop sizes its reserve and text K/V cache for: with <c>numerics.wanF16</c> on, F16 only
/// when no forward of the loop can take the F32 path, so per-frame timesteps, the step cache, an A14B
/// expert pair and a CFG worker without F16 activations each keep it F32; and the loop holds the switches it sized for.</summary>
public sealed class WanDenoiseActivationTests
{
    [Theory]
    [InlineData("plain", true)]
    [InlineData("per-frame timesteps", false)]
    [InlineData("context parallelism", true)]
    [InlineData("step cache", false)]
    [InlineData("A14B pair", false)]
    [InlineData("CFG worker without F16", false)]
    public void TheLoopSizesForF16OnlyWhenNoForwardCanTakeTheF32Path(string path, bool f16)
    {
        WanVideoConfig cfg = path == "A14B pair" ? WanVideoConfig.T2V_14B with { BoundaryRatio = 0.875f } : WanVideoConfig.Ti2V5B;
        using WanVideoTransformer transformer = new(cfg);
        AttentionTraceBackend backend = new() { SupportsF16Activations = true };
        IBackend? cfgWorker = path == "CFG worker without F16" ? new AttentionTraceBackend() : null;
        using IDisposable knobs = KnobProfile.Create("wan-f16-loop").With(EngineKnobs.WanF16, true).With(EngineKnobs.DitF16, true).Push();

        DType act = WanVideoPipeline.DenoiseActivation(transformer, backend, cfgWorker,
            stepCache: path == "step cache",
            singleGroup: path != "per-frame timesteps");

        Assert.Equal(f16 ? DType.F16 : DType.F32, act);
    }

    /// <summary>The loop runs under the switches it sized for: a settings change mid-generation does not reach its forwards, and the
    /// request's own settings still apply.</summary>
    [Fact]
    public void TheLoopKeepsTheSwitchesItWasSizedForOverTheRequestsSettings()
    {
        try
        {
            KnobStore.Set(EngineKnobs.WanF16, true);
            using (KnobProfile.Create("request").With(EngineKnobs.SageAttn, false).Push())
            using (WanVideoPipeline.PinBranchSwitches())
            {
                KnobStore.Set(EngineKnobs.WanF16, false);
                Assert.True(EngineKnobs.WanF16.Value);
                Assert.False(EngineKnobs.SageAttn.Value);
            }
            Assert.False(EngineKnobs.WanF16.Value);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.WanF16);
        }
    }
}
