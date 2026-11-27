using System.Globalization;

namespace HartsyInference.LLM.Ernie5;

/// <summary>Maps every checkpoint key to the parameter it feeds, for one config. MTP layers (indices from <c>num_hidden_layers</c>) map only
/// when asked for and are skipped otherwise; every other key must feed a parameter, and every parameter must have a key.</summary>
public static class Ernie5WeightMap
{
    /// <summary>Every parameter key the main model reads for <paramref name="config"/>, scale companions and MTP layers excluded.</summary>
    public static IReadOnlyList<string> ExpectedKeys(Ernie5Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        List<string> keys =
        [
            Ernie5WeightNames.Embedding, Ernie5WeightNames.FinalNorm, Ernie5WeightNames.LmHead, Ernie5WeightNames.HeadFn,
            Ernie5WeightNames.HeadBase, Ernie5WeightNames.HeadScale,
        ];
        if (config.Multimax is not null) keys.AddRange([Ernie5WeightNames.MultimaxRanges, Ernie5WeightNames.MultimaxGains]);
        for (int layer = 0; layer < config.NumHiddenLayers; layer++) AddBlockKeys(config, layer, keys);
        return keys;
    }

    /// <summary>Every parameter key of the multi-token-prediction layers, scale companions excluded: each layer's own leaves
    /// (<see cref="Ernie5WeightNames.MtpLeaves"/>) and its decoder block. The embedding and LM head are the main model's.</summary>
    public static IReadOnlyList<string> ExpectedMtpKeys(Ernie5Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        List<string> keys = [];
        for (int layer = config.NumHiddenLayers; layer < config.TotalLayers; layer++)
        {
            foreach (string leaf in Ernie5WeightNames.MtpLeaves) keys.Add(Ernie5WeightNames.Mtp(layer, leaf));
            AddBlockKeys(config, layer, keys);
        }
        return keys;
    }

    /// <summary>Classifies every key of a checkpoint (raw safetensors keys, scale companions included, or logical weight names). MTP keys map
    /// when <paramref name="mtp"/> and are counted as skipped otherwise.</summary>
    public static Ernie5WeightCoverage Check(Ernie5Config config, IEnumerable<string> keys, bool mtp = false)
    {
        ArgumentNullException.ThrowIfNull(keys);
        HashSet<string> expected = new(ExpectedKeys(config), StringComparer.Ordinal);
        if (mtp) expected.UnionWith(ExpectedMtpKeys(config));
        HashSet<string> seen = new(StringComparer.Ordinal);
        List<string> unmapped = [];
        int mapped = 0, companions = 0, skipped = 0;
        foreach (string key in keys)
        {
            if (!mtp && IsMtpKey(config, key))
            {
                skipped++;
                continue;
            }
            bool companion = key.EndsWith(Ernie5WeightNames.ScaleSuffix, StringComparison.Ordinal);
            string weight = companion ? key[..^Ernie5WeightNames.ScaleSuffix.Length] : key;
            if (!expected.Contains(weight))
            {
                unmapped.Add(key);
                continue;
            }
            if (companion)
            {
                companions++;
                continue;
            }
            mapped++;
            seen.Add(key);
        }
        List<string> missing = [.. expected.Where(k => !seen.Contains(k))];
        return new Ernie5WeightCoverage(mapped, companions, skipped, unmapped, missing);
    }

    /// <summary>Whether <paramref name="key"/> belongs to a multi-token-prediction layer, which follows the decoder layers.</summary>
    public static bool IsMtpKey(Ernie5Config config, string key)
    {
        ArgumentNullException.ThrowIfNull(config);
        const string prefix = "model.layers.";
        if (!key.StartsWith(prefix, StringComparison.Ordinal)) return false;
        int end = key.IndexOf('.', prefix.Length);
        if (end < 0 || !int.TryParse(key.AsSpan(prefix.Length, end - prefix.Length), NumberStyles.None, CultureInfo.InvariantCulture, out int layer))
            return false;
        return config.IsMtpLayer(layer);
    }

    // The decoder block of a main or MTP layer: norms, both hyper-connections, the attention and the FFN.
    private static void AddBlockKeys(Ernie5Config config, int layer, List<string> keys)
    {
        string prefix = Ernie5WeightNames.Layer(layer);
        keys.Add($"{prefix}.{Ernie5WeightNames.AttnNorm}");
        keys.Add($"{prefix}.{Ernie5WeightNames.FfnNorm}");
        foreach (string module in (string[])[Ernie5WeightNames.AttnHc, Ernie5WeightNames.FfnHc])
            foreach (string leaf in Ernie5WeightNames.HcLeaves) keys.Add(Ernie5WeightNames.Hc(config, layer, module, leaf));
        foreach (string leaf in config.IsDsaLayer(layer) ? Ernie5WeightNames.DsaLeaves : Ernie5WeightNames.HcaLeaves)
            keys.Add(Ernie5WeightNames.Attn(layer, leaf));
        if (config.UseGatedAttn) keys.Add(Ernie5WeightNames.Attn(layer, Ernie5WeightNames.AttnGate));
        if (config.UseVhaAttention) foreach (string leaf in Ernie5WeightNames.VhaLeaves) keys.Add(Ernie5WeightNames.Attn(layer, leaf));
        if (layer < config.FirstKDenseReplace)
            foreach (string leaf in Ernie5WeightNames.DenseFfnLeaves) keys.Add(Ernie5WeightNames.DenseFfn(layer, leaf));
        else
            keys.AddRange(Ernie5WeightNames.MoeKeys(config, layer).Select(k => k.Checkpoint));
    }
}
