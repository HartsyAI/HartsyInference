using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The LongCat recipes encode prompts on the denoiser's backend, so a separate text-encoder backend is refused clearly, not ignored.</summary>
public sealed class LongCatTextEncoderBackendTests
{
    [Fact]
    public void ASeparateTextEncoderBackendIsRefusedBeforeAnyWeightIsRead()
    {
        using CpuBackend denoiser = new();
        using CpuBackend textEncoder = new();
        RecipeContext context = new() { CheckpointPath = "no-such-folder", Backend = denoiser, TextEncoderBackend = textEncoder };

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() =>
            LongCatRecipeSupport.RefuseSeparateTextEncoder(context));
        Assert.Contains("TextEncoderBackend", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheDenoisersOwnBackendOrNoneIsAccepted()
    {
        using CpuBackend denoiser = new();
        LongCatRecipeSupport.RefuseSeparateTextEncoder(new RecipeContext { CheckpointPath = "x", Backend = denoiser });
        LongCatRecipeSupport.RefuseSeparateTextEncoder(new RecipeContext { CheckpointPath = "x", Backend = denoiser, TextEncoderBackend = denoiser });
    }
}
