using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;
using Kind = HartsyInference.Diffusion.Pipelines.LongCat.LongCatAvatarSchedule.Kind;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The avatar sigma lists on hand-checked values: the linear ramp and the two distilled paths (50 samples for avatar 1.0 and 1.6, 8 for
/// 1.5). The reference comparison is <c>LongCatAvatarScheduleParityTests</c>.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatAvatarScheduleTests
{
    [Fact]
    public void LinearRampMatchesTorchLinspaceWithoutATerminalZero()
    {
        float[] sigmas = LongCatAvatarSchedule.Sigmas(4, Kind.Linear);

        // torch.linspace(1, 0.001, 4): 1, 0.667, 0.334, 0.001 (step -0.333).
        Assert.Equal(new[] { 1f, 0.667f, 0.334f, 0.001f }, sigmas, new FloatComparer(1e-5f));
    }

    [Fact]
    public void Distill50SamplesTheFlippedMultiplesOfTwenty()
    {
        // Four steps of the 50-sample path: floor(linspace(0, 50, 4, endpoint=False)) = 0, 12, 25, 37 → (50−k)·20/1000.
        float[] sigmas = LongCatAvatarSchedule.Sigmas(4, Kind.Distill50);

        Assert.Equal(new[] { 1f, 0.76f, 0.5f, 0.26f }, sigmas, new FloatComparer(1e-6f));
    }

    [Fact]
    public void Distill8IsEightDescendingSigmasWithTheFirstAtOne()
    {
        float[] sigmas = LongCatAvatarSchedule.Sigmas(8, Kind.Distill8);

        Assert.Equal(new[] { 999f, 874f, 749f, 624f, 499f, 374f, 249f, 124f }.Select(v => v / 999f).ToArray(), sigmas,
            new FloatComparer(1e-6f));
    }

    [Fact]
    public void Distill8RefusesAnyStepCountOtherThanEight()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => LongCatAvatarSchedule.Sigmas(4, Kind.Distill8));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void EveryPathRefusesANonPositiveStepCount(int steps)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => LongCatAvatarSchedule.Sigmas(steps, Kind.Linear));
    }

    private sealed class FloatComparer(float tolerance) : IEqualityComparer<float>
    {
        public bool Equals(float x, float y) => Math.Abs(x - y) <= tolerance;
        public int GetHashCode(float value) => value.GetHashCode();
    }
}
