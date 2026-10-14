using HartsyInference.Audio.Sampling;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>CPU-only checks for the shared end-of-generation gate. The gate must keep stop ids out of the draw
/// until its floor is reached, and must be a no-op once the floor is passed.</summary>
public sealed class StopGateTests
{
    [Fact]
    public void Floor_IsMaxOfAbsoluteAndRatio()
    {
        StopGate gate = new(MinTokens: 11, MinTokenTextRatio: 2f);
        Assert.Equal(11, gate.FloorFor(targetTextTokens: 3));    // absolute floor wins
        Assert.Equal(20, gate.FloorFor(targetTextTokens: 10));   // ratio wins
    }

    [Fact]
    public void Disabled_NeverMasks()
    {
        float[] logits = [0f, 1f, 2f];
        StopGate.Disabled.Apply(logits, generated: 0, targetTextTokens: 0, stopId: 2);
        Assert.Equal(2f, logits[2]);
    }

    [Fact]
    public void ClosedGate_MasksEveryStopId_AndOpensExactlyAtFloor()
    {
        StopGate gate = new(MinTokens: 3);
        int[] stops = [0, 2];

        float[] closedAtFloorMinusOne = [5f, 1f, 5f, 1f];
        gate.Apply(closedAtFloorMinusOne, generated: 2, targetTextTokens: 0, stops);
        Assert.Equal(float.NegativeInfinity, closedAtFloorMinusOne[0]);
        Assert.Equal(float.NegativeInfinity, closedAtFloorMinusOne[2]);
        Assert.Equal(1f, closedAtFloorMinusOne[1]);

        float[] openAtFloor = [5f, 1f, 5f, 1f];
        gate.Apply(openAtFloor, generated: 3, targetTextTokens: 0, stops);
        Assert.Equal(5f, openAtFloor[0]);
        Assert.Equal(5f, openAtFloor[2]);
    }

    [Fact]
    public void StopIdOutsideHead_IsUnreachable_SoNothingIsMasked()
    {
        float[] logits = [0f, 3f];
        StopGate.Suppress(logits, [2, 999_999]);
        Assert.Equal(0f, logits[0]);
        Assert.Equal(3f, logits[1]);
    }
}
