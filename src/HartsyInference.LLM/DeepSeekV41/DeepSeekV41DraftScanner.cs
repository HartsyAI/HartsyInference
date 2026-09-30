using System.Globalization;
using HartsyInference.ModelAssets.Quant;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Decides from key names alone which draft routed experts have every tensor their producer's quantization needs.</summary>
internal static class DeepSeekV41DraftScanner
{
    private const string DraftPrefix = "mtp.";

    /// <summary>The draft report, plus every key of an incomplete expert so the caller can keep them out of companion binding.</summary>
    internal static (DeepSeekV41DraftReport Report, HashSet<string> IncompleteKeys) Scan(
        IReadOnlySet<string> canonicalKeys, DeepSeekV41Config config, QuantFlavor flavor)
    {
        HashSet<string> incompleteKeys = new(StringComparer.Ordinal);
        Dictionary<int, IReadOnlyList<int>> missing = new();
        string[] draftKeys = canonicalKeys.Where(static key => key.StartsWith(DraftPrefix, StringComparison.Ordinal)).ToArray();
        if (draftKeys.Length == 0)
            return (new DeepSeekV41DraftReport(DeepSeekV41DraftStatus.Absent, missing, config.DsparkNRoutedExperts), incompleteKeys);

        string[] leaves = ExpertLeaves(flavor);
        for (int draft = 0; draft < config.NumNextnPredictLayers; draft++)
        {
            List<int> lacking = new();
            for (int expert = 0; expert < config.DsparkNRoutedExperts; expert++)
            {
                if (!ExpertComplete(canonicalKeys, draft, expert, leaves))
                    lacking.Add(expert);
            }
            if (lacking.Count > 0)
                missing[draft] = lacking;
            foreach (int expert in lacking)
            {
                string prefix = string.Create(CultureInfo.InvariantCulture, $"{DraftPrefix}{draft}.ffn.experts.{expert}.");
                foreach (string key in draftKeys.Where(key => key.StartsWith(prefix, StringComparison.Ordinal)))
                    incompleteKeys.Add(key);
            }
        }
        DeepSeekV41DraftStatus status = missing.Count == 0 ? DeepSeekV41DraftStatus.Complete : DeepSeekV41DraftStatus.Incomplete;
        return (new DeepSeekV41DraftReport(status, missing, config.DsparkNRoutedExperts), incompleteKeys);
    }

    private static bool ExpertComplete(IReadOnlySet<string> keys, int draft, int expert, string[] leaves)
    {
        foreach (string projection in new[] { "w1", "w2", "w3" })
        {
            foreach (string leaf in leaves)
            {
                string key = string.Create(CultureInfo.InvariantCulture, $"{DraftPrefix}{draft}.ffn.experts.{expert}.{projection}.{leaf}");
                if (!keys.Contains(key))
                    return false;
            }
        }
        return true;
    }

    private static string[] ExpertLeaves(QuantFlavor flavor) => flavor switch
    {
        QuantFlavor.Official => ["weight", "scale"],
        QuantFlavor.AmdQuark => ["weight", "weight_scale"],
        QuantFlavor.NvidiaNvfp4 => ["weight", "weight_scale", "weight_scale_2"],
        QuantFlavor.Mlx => ["weight", "scales", "biases"],
        QuantFlavor.Exl3 => ["trellis", "suh", "svh", "mcg"],
        _ => throw new ArgumentOutOfRangeException(nameof(flavor), flavor, null),
    };
}
