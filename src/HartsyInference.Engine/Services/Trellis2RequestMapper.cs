using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Pipelines.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Maps a mesh request onto TRELLIS.2's options, and rejects the texture options for every model that has no texture output.</summary>
internal static class Trellis2RequestMapper
{
    /// <summary>The TRELLIS.2 options of a request; a negative request seed draws a random one.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The seed does not fit the sampler's 32-bit seed.</exception>
    public static Trellis2Options ToOptions(MeshRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Seed > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(request), request.Seed, $"The TRELLIS.2 seed must be at most {int.MaxValue}.");
        return new Trellis2Options
        {
            Grid = request.GridResolution,
            Steps = request.Steps,
            Seed = request.Seed < 0 ? null : (int)request.Seed,
            Texture = request.Texture,
            TextureSize = request.TextureSize,
        };
    }

    /// <summary>Fails when a texture option is set for a model that has no texture output, so the request is never silently changed.
    /// TRELLIS.2 and Pixal3D texture, so the caller skips this check for them.</summary>
    /// <exception cref="NotSupportedException">Texture output, or a texture size, on a model that has none.</exception>
    public static void RejectForOtherModels(MeshRequest request, string modelId)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Texture || request.TextureSize != 0)
            throw new NotSupportedException($"'{modelId}' has no texture output; Texture and TextureSize apply to trellis2 and {MeshModelIds.Pixal3D}.");
    }
}
