namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Assigns a canonical DeepSeek-V4.1 tensor key to its <see cref="DeepSeekV41WeightClass"/> from the name alone.</summary>
public static class DeepSeekV41WeightClassifier
{
    private const string LayerPrefix = "layers.";

    /// <summary>The class of <paramref name="canonicalKey"/>, or null when the name is not a DeepSeek-V4.1 tensor.</summary>
    public static DeepSeekV41WeightClass? Classify(string canonicalKey)
    {
        ArgumentNullException.ThrowIfNull(canonicalKey);
        if (canonicalKey.StartsWith("mtp.", StringComparison.Ordinal))
            return DeepSeekV41WeightClass.Draft;
        if (canonicalKey.StartsWith("vision.", StringComparison.Ordinal)
            || canonicalKey.StartsWith("aligner.", StringComparison.Ordinal)
            || canonicalKey is "image_start" or "image_end" or "image_newline")
        {
            return DeepSeekV41WeightClass.Vision;
        }
        if (canonicalKey.StartsWith("embed.", StringComparison.Ordinal))
            return DeepSeekV41WeightClass.Embed;
        if (canonicalKey.StartsWith("head.", StringComparison.Ordinal) || canonicalKey.StartsWith("norm.", StringComparison.Ordinal))
            return DeepSeekV41WeightClass.Head;
        return ClassifyLayerKey(canonicalKey);
    }

    private static DeepSeekV41WeightClass? ClassifyLayerKey(string key)
    {
        if (!key.StartsWith(LayerPrefix, StringComparison.Ordinal))
            return null;
        int digitsEnd = LayerPrefix.Length;
        while (digitsEnd < key.Length && char.IsAsciiDigit(key[digitsEnd]))
            digitsEnd++;
        if (digitsEnd == LayerPrefix.Length || digitsEnd >= key.Length || key[digitsEnd] != '.')
            return null;
        ReadOnlySpan<char> rest = key.AsSpan(digitsEnd + 1);
        if (rest.StartsWith("ffn.experts.", StringComparison.Ordinal))
            return DeepSeekV41WeightClass.Expert;
        if (rest.StartsWith("engram.embed.", StringComparison.Ordinal))
            return DeepSeekV41WeightClass.Engram;
        return DeepSeekV41WeightClass.Dense;
    }
}
