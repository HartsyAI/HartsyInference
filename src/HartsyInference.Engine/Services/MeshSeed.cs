namespace HartsyInference.Engine.Services;

/// <summary>Resolves a request's seed to the 32-bit seed a sampler takes, so the seed a result reports is the one that ran.</summary>
internal static class MeshSeed
{
    /// <summary>The seed a run uses for <paramref name="requested"/>: a negative request draws one from <c>[0, int.MaxValue)</c>.</summary>
    /// <param name="requested">The request's seed.</param>
    /// <param name="wrapsLargeSeeds">True for the models that cast a seed above <see cref="int.MaxValue"/> to <c>int</c> unchecked (it wraps to a negative value); the others reject it before this is called.</param>
    public static long Resolve(long requested, bool wrapsLargeSeeds)
    {
        if (requested < 0)
        {
            return Random.Shared.Next(0, int.MaxValue);
        }
        // The sampler gets the low 32 bits as an int. Report them as an unsigned value: sending a reported seed back wraps to the same int, where a negative one would draw a new random seed.
        return wrapsLargeSeeds && requested > int.MaxValue ? requested & 0xFFFFFFFFL : requested;
    }
}
