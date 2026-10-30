using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Anima's preset matches the RoPE extrapolation ComfyUI applies to the single-file checkpoint.</summary>
public sealed class AnimaConfigTests
{
    [Fact]
    public void Preview3_UsesComfyUiRopeExtrapolation()
    {
        // comfy/model_detection.py: rope_t / rope_h / rope_w extrapolation ratio = 1.0 / 4.0 / 4.0 for the 16-channel
        // Cosmos-Predict2 (Anima) DiT. The diffusers Cosmos class default (2.0, 1.0, 1.0) does not match this checkpoint.
        Assert.Equal((1.0f, 4.0f, 4.0f), AnimaConfig.AnimaPreview3.RopeScale);
    }
}
