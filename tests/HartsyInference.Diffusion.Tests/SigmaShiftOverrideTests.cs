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

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(1e39)]
    [InlineData(1e-50)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void RecipeMapper_RefusesABadStaticShiftByTheRequestField(double shift)
    {
        ImageRequest request = new ImageRequest { Prompt = "p", SigmaShift = shift };
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() => RecipeRequestMapper.ToTextToImage(request, ""));
        Assert.Equal(nameof(ImageRequest.SigmaShift), error.ParamName);
    }

    /// <summary>Flux.2 and Krea 2 read the value as mu (ModelSamplingFlux max_shift), where SwarmUI's minimum 0 is a
    /// real setting, so only a negative, non-finite or exp-overflowing mu is refused.</summary>
    [Theory]
    [InlineData(-1.0)]
    [InlineData(1e39)]
    [InlineData(100.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void FluxMaxShift_RefusesABadMuByTheRequestField(double shift)
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() => RecipeRequestMapper.MapFluxMaxShift(shift));
        Assert.Equal(nameof(ImageRequest.SigmaShift), error.ParamName);
    }

    /// <summary>Mu 0 builds the exact schedulers the Flux.2 and Krea 2 pipelines construct from it.</summary>
    [Fact]
    public void FluxMaxShift_ZeroIsAcceptedOnFlux2AndKrea2()
    {
        float mu = Assert.IsType<float>(RecipeRequestMapper.MapFluxMaxShift(0.0));
        Assert.Equal(0f, mu);
        Assert.Null(RecipeRequestMapper.MapFluxMaxShift(null));

        FlowMatchEulerDiscreteScheduler flux2 = FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(4096, maxShift: mu);
        Assert.Equal(1f, flux2.Shift, 5);
        FlowMatchEulerDiscreteScheduler krea2 = new FlowMatchEulerDiscreteScheduler().WithShift(MathF.Exp(mu));
        Assert.Equal(1f, krea2.Shift);
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
