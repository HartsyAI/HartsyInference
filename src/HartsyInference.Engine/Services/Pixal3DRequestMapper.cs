using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Pipelines.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Maps a <see cref="MeshRequest"/> to Pixal3D's options and keeps Pixal3D-only fields from silently changing another model's request.</summary>
public static class Pixal3DRequestMapper
{
    /// <summary>The options of a Pixal3D request.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The seed does not fit the generator's 32 bits.</exception>
    public static Pixal3DOptions ToOptions(MeshRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Seed > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(request), request.Seed, $"The Pixal3D seed must be at most {int.MaxValue}.");
        return new Pixal3DOptions
        {
            Grid = request.GridResolution,
            Steps = request.Steps,
            Seed = request.Seed < 0 ? null : (int)request.Seed,
            FovRadians = request.FieldOfViewRadians,
        };
    }

    /// <summary>Fails when the Pixal3D-only field of view is set for another model, so the request is never silently changed.</summary>
    /// <exception cref="NotSupportedException">A field of view on a model other than pixal3d.</exception>
    public static void RejectForOtherModels(MeshRequest request, string modelId)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.FieldOfViewRadians is not null && !string.Equals(modelId, MeshModelIds.Pixal3D, StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException($"3D model '{modelId}' takes no field of view; FieldOfViewRadians applies to {MeshModelIds.Pixal3D}.");
    }
}
