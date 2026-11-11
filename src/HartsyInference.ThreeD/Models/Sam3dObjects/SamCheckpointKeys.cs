namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// Key layout of the released SAM 3D checkpoints, from the upstream loader (<c>inference_pipeline.py</c>, <c>model/io.py</c>): the Lightning
/// <c>.ckpt</c> generators keep their weights under a <c>state_dict</c> envelope with the prefixes <c>_base_models.generator.</c> and
/// <c>_base_models.condition_embedder.</c>, the decoders have no prefix, and upstream reads a <c>.safetensors</c> with the same prefix filter.
/// A safetensors conversion may keep those prefixes or drop them; both are accepted. Whether the released files follow this layout is
/// unverified until a checkpoint header is read.
/// </summary>
public static class SamCheckpointKeys
{
    public const string GeneratorPrefix = "_base_models.generator.";

    public const string ConditionEmbedderPrefix = "_base_models.condition_embedder.";

    private const string StateDictEnvelope = "state_dict.";

    /// <summary>
    /// Maps each wanted tensor's name inside its model (prefix and envelope removed) to its key in the file. A generator file without any
    /// prefixed key is taken to be an already-stripped conversion and yields every key; a condition-embedder request without prefixed keys is empty.
    /// </summary>
    public static Dictionary<string, string> Select(IEnumerable<string> fileKeys, SamCheckpointSection section)
    {
        ArgumentNullException.ThrowIfNull(fileKeys);
        List<(string Source, string Name)> keys = [];
        foreach (string key in fileKeys)
            keys.Add((key, key.StartsWith(StateDictEnvelope, StringComparison.Ordinal) ? key[StateDictEnvelope.Length..] : key));

        Dictionary<string, string> result = new(StringComparer.Ordinal);
        if (section == SamCheckpointSection.Root)
        {
            foreach ((string source, string name) in keys)
                result[name] = source;
            return result;
        }

        string prefix = section == SamCheckpointSection.Generator ? GeneratorPrefix : ConditionEmbedderPrefix;
        bool anyPrefixed = keys.Exists(k => k.Name.StartsWith(GeneratorPrefix, StringComparison.Ordinal)
            || k.Name.StartsWith(ConditionEmbedderPrefix, StringComparison.Ordinal));
        foreach ((string source, string name) in keys)
        {
            if (name.StartsWith(prefix, StringComparison.Ordinal))
                result[name[prefix.Length..]] = source;
            else if (!anyPrefixed && section == SamCheckpointSection.Generator)
                result[name] = source;
        }

        return result;
    }
}
