using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The FLUX.2 and FLUX.3 recipes refuse placement backends their pipelines do not honour, naming each one; the
/// default (null or the primary backend) passes.</summary>
public sealed class UnhonouredBackendsTests
{
    [Fact]
    public void Default_and_primary_equal_backends_pass()
    {
        using CpuBackend primary = new();
        RecipeContext context = Context(primary) with
        {
            TextEncoderBackend = primary, VaeBackend = primary, CfgParallelBackend = null, DitShardBackends = null, CpBackends = null,
        };

        UnhonouredBackends.Refuse(context, "FLUX.2");
    }

    [Fact]
    public void Each_separate_backend_is_named_in_the_refusal()
    {
        using CpuBackend primary = new();
        using CpuBackend second = new();
        using CpuBackend third = new();

        NotSupportedException text = Assert.Throws<NotSupportedException>(() =>
            UnhonouredBackends.Refuse(Context(primary) with { TextEncoderBackend = second }, "FLUX.2"));
        Assert.Contains("TextEncoderBackend", text.Message, StringComparison.Ordinal);

        NotSupportedException vae = Assert.Throws<NotSupportedException>(() =>
            UnhonouredBackends.Refuse(Context(primary) with { VaeBackend = second }, "FLUX.2"));
        Assert.Contains("VaeBackend", vae.Message, StringComparison.Ordinal);

        NotSupportedException all = Assert.Throws<NotSupportedException>(() =>
            UnhonouredBackends.Refuse(Context(primary) with
            {
                TextEncoderBackend = second, VaeBackend = third, CfgParallelBackend = second,
                DitShardBackends = [second], CpBackends = [third],
            }, "FLUX.3"));
        Assert.Contains("TextEncoderBackend, VaeBackend, CfgParallelBackend, DitShardBackends, CpBackends", all.Message, StringComparison.Ordinal);
        Assert.Contains("FLUX.3", all.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Flux2_recipe_refuses_before_loading_anything()
    {
        using CpuBackend primary = new();
        using CpuBackend second = new();

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() =>
            new Flux2Recipe().Construct(Context(primary) with { TextEncoderBackend = second }));

        Assert.Contains("FLUX.2", refused.Message, StringComparison.Ordinal);
        Assert.Contains("TextEncoderBackend", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Flux3_recipe_refuses_before_loading_anything()
    {
        using CpuBackend primary = new();
        using CpuBackend second = new();

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() =>
            new Flux3Recipe().Construct(Context(primary) with { CfgParallelBackend = second }));

        Assert.Contains("FLUX.3", refused.Message, StringComparison.Ordinal);
        Assert.Contains("CfgParallelBackend", refused.Message, StringComparison.Ordinal);
    }

    private static RecipeContext Context(IBackend backend) => new()
    {
        CheckpointPath = "no-such-folder",
        Backend = backend,
    };
}
