using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The FLUX.2 and FLUX.3 recipes refuse the transformer placements they do not honour, and warn about the text-encoder and VAE
/// placements they run on the primary backend. The default (null or the primary backend) passes.</summary>
public sealed class UnhonouredBackendsTests
{
    [Fact]
    public void Default_and_primary_equal_backends_pass_and_the_transformer_stays_on_the_primary()
    {
        using CpuBackend primary = new();
        RecipeContext context = Context(primary) with
        {
            TextEncoderBackend = primary, VaeBackend = primary, CfgParallelBackend = null, DitShardBackends = null, CpBackends = null,
        };

        UnhonouredBackends.Refuse(context, "FLUX.2");

        Assert.Equal([primary], context.TransformerBackends);
    }

    [Fact]
    public void A_separate_text_encoder_or_vae_warns_and_does_not_refuse()
    {
        using CpuBackend primary = new();
        using CpuBackend second = new();

        UnhonouredBackends.Refuse(Context(primary) with { TextEncoderBackend = second }, "FLUX.2");
        UnhonouredBackends.Refuse(Context(primary) with { VaeBackend = second }, "FLUX.3");
    }

    [Fact]
    public void A_separate_cfg_parallel_backend_is_refused_by_name()
    {
        using CpuBackend primary = new();
        using CpuBackend second = new();

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() =>
            UnhonouredBackends.Refuse(Context(primary) with { CfgParallelBackend = second }, "FLUX.2"));

        Assert.Contains("CfgParallelBackend", refused.Message, StringComparison.Ordinal);
        Assert.Contains("--cfg-parallel-gpu", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void A_separate_singular_dit_shard_backend_is_refused_by_name()
    {
        using CpuBackend primary = new();
        using CpuBackend second = new();

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() =>
            UnhonouredBackends.Refuse(Context(primary) with { DitShardBackend = second }, "FLUX.2"));

        Assert.Contains("DitShardBackend", refused.Message, StringComparison.Ordinal);
        Assert.Contains("--dit-shard-gpu", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void A_separate_dit_shard_list_and_context_parallel_list_are_refused_by_name()
    {
        using CpuBackend primary = new();
        using CpuBackend second = new();

        NotSupportedException shard = Assert.Throws<NotSupportedException>(() =>
            UnhonouredBackends.Refuse(Context(primary) with { DitShardBackends = [primary, second] }, "FLUX.2"));
        Assert.Contains("DitShardBackends", shard.Message, StringComparison.Ordinal);

        NotSupportedException context = Assert.Throws<NotSupportedException>(() =>
            UnhonouredBackends.Refuse(Context(primary) with { CpBackends = [primary, second] }, "FLUX.2"));
        Assert.Contains("CpBackends", context.Message, StringComparison.Ordinal);
        Assert.Contains("--cp-gpu", context.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Every_refused_transformer_placement_is_named_in_one_message()
    {
        using CpuBackend primary = new();
        using CpuBackend second = new();
        using CpuBackend third = new();

        NotSupportedException all = Assert.Throws<NotSupportedException>(() =>
            UnhonouredBackends.Refuse(Context(primary) with
            {
                TextEncoderBackend = second, VaeBackend = third, CfgParallelBackend = second,
                DitShardBackend = third, DitShardBackends = [second], CpBackends = [third],
            }, "FLUX.3"));

        Assert.Contains("FLUX.3", all.Message, StringComparison.Ordinal);
        Assert.Contains("CfgParallelBackend, DitShardBackend, DitShardBackends, CpBackends", all.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Flux2_recipe_refuses_a_cfg_parallel_placement_before_loading_anything()
    {
        using CpuBackend primary = new();
        using CpuBackend second = new();

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() =>
            new Flux2Recipe().Construct(Context(primary) with { CfgParallelBackend = second }));

        Assert.Contains("FLUX.2", refused.Message, StringComparison.Ordinal);
        Assert.Contains("CfgParallelBackend", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Flux2_recipe_only_warns_for_a_separate_text_encoder()
    {
        using CpuBackend primary = new();
        using CpuBackend second = new();

        // The text encoder is a warning, so construction goes on to the checkpoint, which this context does not have.
        Exception error = Assert.ThrowsAny<Exception>(() =>
            new Flux2Recipe().Construct(Context(primary) with { TextEncoderBackend = second }));
        Assert.IsNotType<NotSupportedException>(error);
    }

    [Fact]
    public void Flux3_recipe_refuses_a_cfg_parallel_placement_before_loading_anything()
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
