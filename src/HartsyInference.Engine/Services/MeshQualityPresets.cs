using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Turns a <see cref="MeshRequest.Quality"/> into concrete settings in one place, before validation, planning and generation, so all three see the same request. A field the request already sets is kept.</summary>
internal static class MeshQualityPresets
{
    /// <summary>The request with the preset's values in every field it left at 0; the request itself when it asks for <see cref="MeshQuality.Default"/> or the model has no such preset (the validator rejects that case).</summary>
    public static MeshRequest Resolve(string id, MeshRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Quality == MeshQuality.Default || MeshModelProfiles.Find(id)?.Presets.GetValueOrDefault(request.Quality) is not MeshPreset preset)
            return request;
        return Apply(preset, request, TakesRemesh(id));
    }

    /// <summary>The request with <paramref name="preset"/>'s values in every field it left at 0.</summary>
    internal static MeshRequest Apply(MeshPreset preset, MeshRequest request, bool remeshCapable = false) => request with
    {
        Steps = request.Steps == 0 ? preset.Steps : request.Steps,
        GridResolution = request.GridResolution == 0 ? PresetGrid(preset, request, remeshCapable) : request.GridResolution,
        TextureSize = request.TextureSize == 0 && request.Texture ? preset.TextureSize : request.TextureSize,
    };

    /// <summary>The value the preset gives <paramref name="field"/> of a request that leaves it unset, or null when it gives none.</summary>
    public static int? Supplied(MeshPreset? preset, MeshRequest request, string field, bool remeshCapable = false)
    {
        int value = field switch
        {
            nameof(MeshRequest.Steps) => preset?.Steps ?? 0,
            nameof(MeshRequest.GridResolution) => preset is null ? 0 : PresetGrid(preset, request, remeshCapable),
            nameof(MeshRequest.TextureSize) => request.Texture ? preset?.TextureSize ?? 0 : 0,
            _ => 0,
        };
        return value == 0 ? null : value;
    }

    // The remesh does not run at 1536 (it needs more host memory than a 12 GB machine has), so a preset's 1536 gives way to 1024 for a request that asks for it.
    private static int PresetGrid(MeshPreset preset, MeshRequest request, bool remeshCapable) => remeshCapable && request.Remesh == true && preset.GridResolution > 1024 ? 1024 : preset.GridResolution;

    /// <summary>Whether the model's profile takes <see cref="MeshRequest.Remesh"/>; only those presets give way to the remesh's grid limit.</summary>
    internal static bool TakesRemesh(string id) =>
        MeshModelProfiles.Find(id)?.Fields.Any(f => f.Name == nameof(MeshRequest.Remesh) && f.Access == HartsyInference.Engine.Planning.MeshFieldAccess.Used) == true;
}
