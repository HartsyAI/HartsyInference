using HartsyInference.Audio.Models.Haru;
using Xunit;

namespace HartsyInference.Audio.Tests.Haru;

/// <summary>Replays the random draws an upstream run made, in order; a draw of another size means the call order drifted.</summary>
internal sealed class HaruScriptedNoise(List<float[]> draws) : IHaruNoise
{
    private int _next;

    internal int Remaining => draws.Count - _next;

    public void FillGaussian(Span<float> destination)
    {
        float[] draw = draws[_next++];
        Assert.Equal(draw.Length, destination.Length);
        draw.CopyTo(destination);
    }
}
