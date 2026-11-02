using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Zero on both sides is a no-op for resolution and for cache identity, through one predicate, so a zeroed LoRA lands on the
/// base pipeline. A section-confined entry is refused by name, since its scope is not modeled and skipping it would hide the gap.</summary>
public sealed class LoraNoOpAndConfinementTests
{
    [Fact]
    public void IsNoOp_OnlyWhenZeroOnBothSides()
    {
        Assert.True(LoraResolver.IsNoOp(new LoraEntry { Model = "a", Weight = 0 }));
        Assert.True(LoraResolver.IsNoOp(new LoraEntry { Model = "a", Weight = 0, TextEncoderWeight = 0 }));
        Assert.False(LoraResolver.IsNoOp(new LoraEntry { Model = "a", Weight = 0, TextEncoderWeight = 0.5 }));
        Assert.False(LoraResolver.IsNoOp(new LoraEntry { Model = "a", Weight = 0.5 }));
    }

    [Fact]
    public void Resolve_DropsANoOpEntryWithoutLookingUpItsFile()
    {
        LoraStack stack = new() { Entries = [new LoraEntry { Model = "not-on-disk-by-design", Weight = 0 }] };

        Assert.Empty(LoraResolver.Resolve(stack));
    }

    [Fact]
    public void Resolve_RefusesASectionConfinedEntryByName()
    {
        LoraStack stack = new() { Entries = [new LoraEntry { Model = "scoped-lora", Weight = 1, SectionConfinement = "2" }] };

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() => LoraResolver.Resolve(stack));

        Assert.Contains("scoped-lora", refused.Message, StringComparison.Ordinal);
        Assert.Contains("section 2", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Resolve_RefusesANonFiniteStrengthByName()
    {
        LoraStack stack = new() { Entries = [new LoraEntry { Model = "broken-strength", Weight = double.NaN }] };

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() => LoraResolver.Resolve(stack));

        Assert.Contains("broken-strength", refused.Message, StringComparison.Ordinal);
        Assert.Contains("finite", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void CacheKey_ZeroedLora_KeysLikeARequestWithNoLora()
    {
        ImageRequest plain = new() { Prompt = "p" };
        ImageRequest zeroed = new()
        {
            Prompt = "p",
            Loras = new LoraStack { Entries = [new LoraEntry { Model = "scoped-lora", Weight = 0 }] },
        };

        Assert.Equal(RecipeCacheKey.Describe(plain), RecipeCacheKey.Describe(zeroed));
    }

    [Fact]
    public void CacheKey_EncoderOnlyLora_StillSplitsTheCache()
    {
        ImageRequest plain = new() { Prompt = "p" };
        ImageRequest encoderOnly = new()
        {
            Prompt = "p",
            Loras = new LoraStack { Entries = [new LoraEntry { Model = "x", Weight = 0, TextEncoderWeight = 0.5 }] },
        };

        Assert.NotEqual(RecipeCacheKey.Describe(plain), RecipeCacheKey.Describe(encoderOnly));
    }
}
