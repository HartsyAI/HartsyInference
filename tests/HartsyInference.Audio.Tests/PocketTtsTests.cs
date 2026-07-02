using HartsyInference.Audio.Models.PocketTts;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for the PocketTTS config skeleton (continuous-latent TTS; exact dims are checkpoint-gated).</summary>
public sealed class PocketTtsTests
{
    [Fact]
    public void Default_DocumentsContinuousLatentStructure()
    {
        PocketTtsConfig c = PocketTtsConfig.Default;
        Assert.Equal(24_000, c.SampleRate);
        Assert.Equal(26, c.Voices.Count);
        Assert.Contains("alba", c.Voices);
        Assert.Equal(1_024, c.DModel);    // flow_lm.transformer.d_model (reconciled from checkpoint)
        Assert.Equal(32, c.LatentDim);    // mimi.inner_dim continuous latent (reconciled from checkpoint)
    }
}
