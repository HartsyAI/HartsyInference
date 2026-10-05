using System.Globalization;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Models.Kokoro;

/// <summary>A Kokoro voice spec naming one voice pack or a blend of several. The reference <c>KPipeline.load_voice</c>
/// takes <c>"af_bella,af_sky"</c> and averages the packs; a part may also carry a weight, <c>"af_bella(2),af_sky(1)"</c>
/// or <c>"af_bella:0.7,af_sky:0.3"</c> (the blend is the weighted mean, so equal weights give the reference's
/// average), and <c>+</c> separates parts as well as <c>,</c>. The first part's voice decides the language.</summary>
public sealed partial class KokoroVoiceMix
{
    /// <summary>The voices and their weights, in the order written.</summary>
    public IReadOnlyList<(string Voice, float Weight)> Parts { get; }

    private KokoroVoiceMix(IReadOnlyList<(string, float)> parts) => Parts = parts;

    /// <summary>The voice whose language the blend speaks: the first part's.</summary>
    public string PrimaryVoice => Parts[0].Voice;

    /// <summary>Whether this names more than one voice.</summary>
    public bool IsBlend => Parts.Count > 1;

    /// <summary>Parses <paramref name="spec"/>.</summary>
    /// <exception cref="ArgumentException">A part is not a voice name (letters, digits, <c>_</c>, <c>-</c>) with an
    /// optional positive weight.</exception>
    public static KokoroVoiceMix Parse(string spec)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(spec);
        List<(string, float)> parts = [];
        foreach (string raw in spec.Split([',', '+'], StringSplitOptions.TrimEntries))
        {
            Match m = PartRegex().Match(raw);
            if (!m.Success)
                throw new ArgumentException($"Kokoro voice '{raw}' in '{spec}' is not a voice name with an optional "
                    + "weight, such as af_bella, af_bella(2) or af_bella:0.5.", nameof(spec));
            string w = m.Groups["w1"].Success ? m.Groups["w1"].Value : m.Groups["w2"].Value;
            float weight = w.Length == 0 ? 1f : float.Parse(w, NumberStyles.Float, CultureInfo.InvariantCulture);
            if (!(weight > 0f) || float.IsInfinity(weight))
                throw new ArgumentException($"Kokoro voice weight in '{raw}' must be a positive number.", nameof(spec));
            parts.Add((m.Groups["name"].Value, weight));
        }
        return new KokoroVoiceMix(parts);
    }

    /// <summary>Blends <paramref name="packs"/> (one per part, in order) into one pack named
    /// <paramref name="name"/>: the weighted mean of each style row.</summary>
    public KokoroVoicePack Blend(string name, IReadOnlyList<KokoroVoicePack> packs)
    {
        ArgumentNullException.ThrowIfNull(packs);
        if (packs.Count != Parts.Count)
            throw new ArgumentException($"Expected {Parts.Count} voice packs, got {packs.Count}.", nameof(packs));
        return KokoroVoicePack.WeightedMean(name, packs, Parts.Select(static p => p.Weight).ToArray());
    }

    [GeneratedRegex(@"^(?<name>[A-Za-z0-9_\-]+)(?:\((?<w1>[0-9.eE+\-]+)\)|:(?<w2>[0-9.eE+\-]+))?$")]
    private static partial Regex PartRegex();
}
