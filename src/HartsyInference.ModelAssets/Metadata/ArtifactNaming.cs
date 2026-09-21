using System.Text;

namespace HartsyInference.ModelAssets.Metadata;

/// <summary>The canonical file name for a published artifact: <c>&lt;engine-id&gt;[-&lt;variant&gt;]_&lt;precision&gt;.&lt;ext&gt;</c>.
///
/// <para>Dashes inside the name, one underscore before the precision token. That is what the published files we
/// already consume mostly read like (<c>krea2_turbo_fp8_scaled</c>, <c>flux2-dev-Q4_K_S</c>) — the convention
/// existed but was never written down, so every new file picked a separator by hand.</para></summary>
public static class ArtifactNaming
{
    /// <summary>Precision token for a file whose weights were not re-quantized, only moved to a new container.</summary>
    public const string RepackPrecision = "repack";

    /// <summary>Builds the canonical name. <paramref name="extension"/> may be given with or without a leading dot.</summary>
    /// <param name="variant">Sub-build such as <c>"turbo"</c> or <c>"dev"</c>; null or empty for a family's only build.</param>
    /// <param name="precision">Token such as <c>"bf16"</c>, <c>"fp8-scaled"</c> or <c>"Q4_K_M"</c>. GGUF preset
    /// names keep their upstream casing, since that is how every published GGUF spells them.</param>
    public static string FileName(string engineId, string? variant, string precision, string extension)
    {
        if (string.IsNullOrWhiteSpace(engineId))
        {
            throw new ArgumentException("An artifact name needs an engine id.", nameof(engineId));
        }
        if (string.IsNullOrWhiteSpace(precision))
        {
            throw new ArgumentException("An artifact name needs a precision token.", nameof(precision));
        }
        StringBuilder name = new(Slug(engineId));
        if (!string.IsNullOrWhiteSpace(variant))
        {
            name.Append('-').Append(Slug(variant));
        }
        name.Append('_').Append(PrecisionToken(precision));
        name.Append(extension.StartsWith('.') ? extension : "." + extension);
        return name.ToString();
    }

    /// <summary>Lowercases and dash-joins a name segment, keeping dots (version numbers such as
    /// <c>qwen-image-2.1</c> are part of the id, not separators).</summary>
    public static string Slug(string value)
    {
        StringBuilder slug = new(value.Length);
        bool lastWasDash = false;
        foreach (char c in value.Trim().ToLowerInvariant())
        {
            if (char.IsAsciiLetterOrDigit(c) || c == '.')
            {
                slug.Append(c);
                lastWasDash = false;
            }
            else if (!lastWasDash && slug.Length > 0)
            {
                slug.Append('-');
                lastWasDash = true;
            }
        }
        return slug.ToString().TrimEnd('-');
    }

    /// <summary>Normalizes a precision token. A GGUF preset (<c>Q4_K_M</c>) keeps its casing and underscores;
    /// anything else is slugged, so <c>"fp8 scaled"</c> and <c>"fp8_scaled"</c> both land on <c>fp8-scaled</c>.</summary>
    public static string PrecisionToken(string precision)
    {
        string trimmed = precision.Trim();
        return IsGgufPreset(trimmed) ? trimmed.ToUpperInvariant() : Slug(trimmed);
    }

    private static bool IsGgufPreset(string value) =>
        value.Length > 1 && (value[0] == 'Q' || value[0] == 'q') && char.IsAsciiDigit(value[1]);
}
