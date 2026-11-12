using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>The one rule for a mesh request's seed: it is a non-negative value that fits the 32-bit seed a sampler takes, or negative for a random draw.</summary>
internal static class MeshSeed
{
    /// <summary>The largest seed a 3D model takes.</summary>
    public const long Max = int.MaxValue;

    /// <summary>Throws when the request's seed is above <see cref="Max"/>; every model's validation calls this, so no model wraps a large seed to a different one.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The seed is above <see cref="Max"/>; attributed to the Seed field.</exception>
    public static void Require(MeshRequest request)
    {
        if (request.Seed > Max)
        {
            throw new ArgumentOutOfRangeException(nameof(request), request.Seed, $"Seed must be at most {Max} (negative draws a random seed); 3D models take a 32-bit seed.")
                .ForField(nameof(MeshRequest.Seed));
        }
    }

    /// <summary>The seed a run uses for <paramref name="requested"/>: a negative request draws one from <c>[0, int.MaxValue)</c>.</summary>
    public static long Resolve(long requested) => requested < 0 ? Random.Shared.Next(0, int.MaxValue) : requested;

    /// <summary>The seed as a sampler takes it: null for a random draw, else the value, which must already be within range.</summary>
    /// <exception cref="OverflowException">The seed is above <see cref="Max"/>, which validation rejects before a run gets here.</exception>
    public static int? ForSampler(long requested) => requested < 0 ? null : checked((int)requested);
}
