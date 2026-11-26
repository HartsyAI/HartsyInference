using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>LongCat's placement policy, through the recipe entry: a CFG-parallel placement is refused by name before any weight is read, and a separate
/// text-encoder backend is accepted (the engine warns and runs it on the primary).</summary>
public sealed class LongCatTextEncoderBackendTests
{
    [Fact]
    public void ACfgParallelPlacementIsRefusedByNameBeforeAnyWeightIsRead()
    {
        using CpuBackend denoiser = new();
        using CpuBackend other = new();
        RecipeContext context = new() { CheckpointPath = "no-such-folder", Backend = denoiser, CfgParallelBackend = other };

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() => new LongCatVideoRecipe().Construct(context));
        Assert.Contains("LongCat", refused.Message, StringComparison.Ordinal);
        Assert.Contains("CfgParallelBackend", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASeparateTextEncoderBackendIsNotRefusedHere()
    {
        using CpuBackend denoiser = new();
        using CpuBackend textEncoder = new();
        RecipeContext context = new() { CheckpointPath = "x", Backend = denoiser, TextEncoderBackend = textEncoder };
        UnhonouredBackends.Refuse(context, "LongCat");
    }
}
