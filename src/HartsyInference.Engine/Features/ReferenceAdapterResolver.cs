using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Features;

/// <summary>Finds the reference adapter a request names, or the released one, and reads its layout. A named adapter must already be on
/// disk; the released default is fetched through the catalog when the server's side-model autofetch allows it.</summary>
public static class ReferenceAdapterResolver
{
    /// <summary>Resolves <paramref name="requested"/> (a file name in the LoRA folders) or, when blank, the released adapter.</summary>
    public static ReferenceAdapterSpec Resolve(string? requested, CancellationToken cancel)
    {
        string path = ModelFileLocator.RequireOrEnsure(requested, SideModels.AnimaRefAdapterV2, "reference adapter", cancel, "Lora");
        return ReferenceAdapterSpec.FromMetadata(path, MetadataJson(path));
    }

    private static string? MetadataJson(string path)
    {
        SafeTensorHeader parsed = SafeTensorHeaderReader.Read(path);
        return parsed.Metadata is { } metadata && metadata.TryGetValue(ReferenceAdapterSpec.MetadataKey, out string? json) ? json : null;
    }
}
