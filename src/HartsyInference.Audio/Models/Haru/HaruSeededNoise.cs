namespace HartsyInference.Audio.Models.Haru;

/// <summary>Standard normal draws from a seeded generator (Box-Muller), so a conversation with a given seed repeats.</summary>
internal sealed class HaruSeededNoise(int seed) : IHaruNoise
{
    private readonly Random _random = new(seed);

    public void FillGaussian(Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = 1.0 - _random.NextDouble();
            destination[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
    }
}
