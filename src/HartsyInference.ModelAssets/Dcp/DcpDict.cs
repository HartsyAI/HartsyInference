namespace HartsyInference.ModelAssets.Dcp;

/// <summary>A pickled dict whose keys may be objects (storage_data is keyed by <c>MetadataIndex</c>), kept as an ordered entry list.</summary>
internal sealed class DcpDict
{
    public List<KeyValuePair<object?, object?>> Entries { get; } = [];
}
