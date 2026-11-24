using System.Globalization;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Maxwell;

/// <summary>Maps every checkpoint key to the parameter it feeds, for one config. MTP layers (indices from <c>num_hidden_layers</c>) map only when
/// asked for and are skipped otherwise; every other key must feed a parameter, and every parameter must have a key.</summary>
public static class MaxwellWeightMap
{
    private const string Model = "Maxwell";

    /// <summary>Every parameter key the main model reads for <paramref name="config"/>, scale companions and MTP layers excluded.</summary>
    public static IReadOnlyList<string> ExpectedKeys(MaxwellConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        List<string> keys = [MaxwellWeightNames.Embedding, MaxwellWeightNames.FinalNorm, MaxwellWeightNames.LmHead];
        foreach (string leaf in MaxwellWeightNames.HeadLeaves) keys.Add(MaxwellWeightNames.Key(MaxwellWeightNames.Head, leaf));
        for (int layer = 0; layer < config.NumHiddenLayers; layer++) AddBlockKeys(config, MaxwellWeightNames.Layer(layer), layer, true, keys);
        return keys;
    }

    /// <summary>Every parameter key of the multi-token-prediction layers: each layer's own leaves and its decoder block, which has dense attention
    /// and no indexer. The embedding and LM head are the main model's.</summary>
    public static IReadOnlyList<string> ExpectedMtpKeys(MaxwellConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        List<string> keys = [];
        for (int mtp = 0; mtp < config.MtpLayers; mtp++)
        {
            foreach (string leaf in MaxwellWeightNames.MtpLeaves) keys.Add(MaxwellWeightNames.Key(MaxwellWeightNames.MtpLayer(mtp), leaf));
            AddBlockKeys(config, MaxwellWeightNames.MtpDecoder(mtp), config.NumHiddenLayers + mtp, false, keys);
        }
        return keys;
    }

    /// <summary>Classifies every key of a checkpoint (raw safetensors keys, scale companions included, or logical weight names). MTP keys map when
    /// <paramref name="mtp"/> and are counted as skipped otherwise.</summary>
    public static CheckpointKeyCoverage Check(MaxwellConfig config, IEnumerable<string> keys, bool mtp = false)
    {
        ArgumentNullException.ThrowIfNull(keys);
        HashSet<string> expected = new(ExpectedKeys(config), StringComparer.Ordinal);
        if (mtp) expected.UnionWith(ExpectedMtpKeys(config));
        HashSet<string> seen = new(StringComparer.Ordinal);
        List<string> unmapped = [];
        int mapped = 0, companions = 0, skipped = 0;
        foreach (string key in keys)
        {
            if (!mtp && IsMtpKey(key))
            {
                skipped++;
                continue;
            }
            bool companion = key.EndsWith(MaxwellWeightNames.ScaleSuffix, StringComparison.Ordinal);
            string weight = companion ? key[..^MaxwellWeightNames.ScaleSuffix.Length] : key;
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
        return new CheckpointKeyCoverage(Model, mapped, companions, skipped, unmapped, missing);
    }

    /// <summary>Whether <paramref name="key"/> belongs to a multi-token-prediction layer.</summary>
    public static bool IsMtpKey(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        const string prefix = "mtp_layers.";
        if (!key.StartsWith(prefix, StringComparison.Ordinal)) return false;
        int end = key.IndexOf('.', prefix.Length);
        return end > 0 && int.TryParse(key.AsSpan(prefix.Length, end - prefix.Length), NumberStyles.None, CultureInfo.InvariantCulture, out _);
    }

    // The decoder block of a main or MTP layer: norms, both hyper-connections, the attention and the FFN.
    private static void AddBlockKeys(MaxwellConfig config, string prefix, int layer, bool indexer, List<string> keys)
    {
        keys.Add(MaxwellWeightNames.Key(prefix, MaxwellWeightNames.AttnNorm));
        keys.Add(MaxwellWeightNames.Key(prefix, MaxwellWeightNames.FfnNorm));
        foreach (string module in (string[])[MaxwellWeightNames.AttnHc, MaxwellWeightNames.FfnHc])
            foreach (string leaf in MaxwellWeightNames.HcLeaves) keys.Add(MaxwellWeightNames.Hc(prefix, module, leaf));
        foreach (string leaf in MaxwellWeightNames.AttnLeaves) keys.Add(MaxwellWeightNames.Key(prefix, $"self_attn.{leaf}"));
        if (indexer)
            foreach (string leaf in MaxwellWeightNames.IndexerLeaves) keys.Add(MaxwellWeightNames.Key(prefix, $"self_attn.indexer.{leaf}"));
        if (config.IsDenseLayer(layer))
        {
            foreach (string leaf in MaxwellWeightNames.FfnLeaves) keys.Add(MaxwellWeightNames.DenseFfn(prefix, leaf));
            return;
        }
        keys.Add(MaxwellWeightNames.Key(prefix, MaxwellWeightNames.Gate));
        keys.Add(MaxwellWeightNames.Key(prefix, MaxwellWeightNames.ExpertBias));
        foreach (string leaf in MaxwellWeightNames.FfnLeaves) keys.Add(MaxwellWeightNames.SharedExpert(prefix, leaf));
        for (int e = 0; e < config.NumExperts; e++)
            foreach (string leaf in MaxwellWeightNames.FfnLeaves) keys.Add(MaxwellWeightNames.Expert(prefix, e, leaf));
    }
}
