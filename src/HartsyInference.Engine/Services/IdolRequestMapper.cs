using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Preprocessing;

namespace HartsyInference.Engine.Services;

/// <summary>Maps an Idol request to the pipeline's options and checks it before any checkpoint loads.</summary>
public static class IdolRequestMapper
{
    /// <summary>The options of an Idol request: steps, guidance, splat count, framing and seed, each at the model's default when the request leaves it unset.</summary>
    /// <exception cref="NotSupportedException">A field Idol does not read is set, or there is no image.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A value is out of the model's range.</exception>
    public static IdolOptions ToOptions(MeshRequest request, string modelId = MeshModelIds.Idol)
    {
        Validate(request, modelId);
        return new IdolOptions
        {
            Steps = request.Steps,
            Guidance = request.Guidance is { } g ? (float)g : null,
            Splats = request.SplatCount,
            Framing = request.Framing == MeshFraming.Photo ? IdolFraming.Photo : IdolFraming.Object,
            Seed = MeshSeed.ForSampler(request.Seed) ?? Random.Shared.Next(),
            Turbo = modelId == MeshModelIds.IdolTurbo,
        };
    }

    /// <summary>Checks an Idol request.</summary>
    public static void Validate(MeshRequest request, string modelId = MeshModelIds.Idol)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (modelId == MeshModelIds.IdolTurbo && request.Guidance is not null)
            throw new NotSupportedException($"'{modelId}' takes no Guidance: the distilled LoRAs carry it and the sampler runs one pass per step.").ForField(nameof(MeshRequest.Guidance));
        if (request.Image is null)
            throw new NotSupportedException("Idol needs MeshRequest.Image, a picture of one object.").ForField(nameof(MeshRequest.Image));
        if (request.GridResolution != 0)
            throw new NotSupportedException("Idol has no grid; GridResolution applies to the mesh models.").ForField(nameof(MeshRequest.GridResolution));
        if (request.FieldOfViewRadians is not null)
            throw new NotSupportedException("Idol takes no field of view; FieldOfViewRadians applies to pixal3d.").ForField(nameof(MeshRequest.FieldOfViewRadians));
        if (request.Steps < 0)
            throw new ArgumentOutOfRangeException(nameof(request), request.Steps, "Steps must not be negative; 0 takes the model's default (50, or 5 for idol-turbo).").ForField(nameof(MeshRequest.Steps));
        if (request.Guidance is { } g && (g < 0 || g > 20 || double.IsNaN(g)))
            throw new ArgumentOutOfRangeException(nameof(request), g, "Guidance must be between 0 and 20.").ForField(nameof(MeshRequest.Guidance));
        IdolOptions probe = new() { Splats = request.SplatCount };
        try
        {
            IdolImageTo3DPipeline.ValidateOptions(probe);
        }
        catch (ArgumentOutOfRangeException error)
        {
            throw error.ForField(nameof(MeshRequest.SplatCount));
        }
    }

    /// <summary>Rejects Idol-only fields on any other model.</summary>
    public static void RejectForOtherModels(MeshRequest request, string modelId)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (MeshModelIds.IsIdol(modelId))
            return;
        if (request.Guidance is not null)
            throw new NotSupportedException($"'{modelId}' takes no Guidance; it applies to {MeshModelIds.Idol}.").ForField(nameof(MeshRequest.Guidance));
        if (request.SplatCount != 0)
            throw new NotSupportedException($"'{modelId}' takes no SplatCount; it applies to {MeshModelIds.Idol}.").ForField(nameof(MeshRequest.SplatCount));
        if (request.Framing != MeshFraming.Object)
            throw new NotSupportedException($"'{modelId}' takes no Framing; it applies to {MeshModelIds.Idol}.").ForField(nameof(MeshRequest.Framing));
    }
}
