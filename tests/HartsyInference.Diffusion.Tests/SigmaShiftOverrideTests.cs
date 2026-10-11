using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>An explicit Sigma Shift reaches the flow-match schedulers and replaces only the shift. A dropped value is
/// silent: the request runs the family's own schedule and nothing says the setting was ignored.</summary>
public sealed class SigmaShiftOverrideTests
{
    [Fact]
    public void WithShift_NullKeepsTheFamilySchedule()
    {
        FlowMatchEulerDiscreteScheduler scheduler = FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(4096);
        Assert.Same(scheduler, scheduler.WithShift(null));
    }

    [Fact]
    public void WithShift_ReplacesADynamicShiftAndKeepsTheShiftTerminal()
    {
        FlowMatchEulerDiscreteScheduler family = FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(
            6889, baseSeqLen: 256, maxSeqLen: 8192, baseShift: 0.5f, maxShift: 0.9f, shiftTerminal: 0.02f);
        FlowMatchEulerDiscreteScheduler overridden = family.WithShift(3.0f);
        FlowMatchEulerDiscreteScheduler expected = new FlowMatchEulerDiscreteScheduler(3.0f, shiftTerminal: 0.02f);

        Assert.Equal(3.0f, overridden.Shift);
        Assert.Equal(expected.SigmasFor(20), overridden.SigmasFor(20));
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void WithShift_RefusesANonPositiveOrNonFiniteShift(float shift)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new FlowMatchEulerDiscreteScheduler(3.0f).WithShift(shift));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BooguFlowMatchScheduler(shift: shift));
    }

    /// <summary>Flux.2 reads the value as SwarmUI's ModelSamplingFlux max_shift: at 4096 tokens the line reaches it, so
    /// the shift is exp(value).</summary>
    [Fact]
    public void DynamicShift_AtTheLinesMaxTokenCountIsExpOfMaxShift()
    {
        FlowMatchEulerDiscreteScheduler scheduler = FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(4096, maxShift: 2.02f);
        Assert.Equal(MathF.Exp(2.02f), scheduler.Shift, 5);
    }

    /// <summary>ComfyUI's Boogu default shift 3.16 is exp(1.15), the line's value at 4096 tokens, so an explicit shift
    /// of exp(1.15) reproduces the family schedule and any other value moves it.</summary>
    [Fact]
    public void BooguShift_MatchesTheLineAtItsOwnValueAndMovesOtherwise()
    {
        BooguFlowMatchScheduler family = new(4096);
        BooguFlowMatchScheduler same = new(4096, shift: MathF.Exp(1.15f));
        BooguFlowMatchScheduler other = new(4096, shift: 6.0f);
        family.SetTimesteps(8);
        same.SetTimesteps(8);
        other.SetTimesteps(8);

        float[] baseline = family.Sigmas();
        float[] matched = same.Sigmas();
        for (int i = 0; i < baseline.Length; i++)
        {
            Assert.Equal(baseline[i], matched[i], 5);
        }
        Assert.True(other.Sigmas()[4] > baseline[4], "a larger shift must keep more noise at mid-schedule");
    }

    [Fact]
    public void RecipeMapper_CarriesTheShiftOntoTheDiffusionRequest()
    {
        ImageRequest request = new ImageRequest { Prompt = "p", SigmaShift = 6.0 };
        TextToImageRequest inner = RecipeRequestMapper.ToTextToImage(request, "");
        Assert.Equal(6.0f, inner.SigmaShift);
        Assert.Null(RecipeRequestMapper.ToTextToImage(request with { SigmaShift = null }, "").SigmaShift);
    }
}
