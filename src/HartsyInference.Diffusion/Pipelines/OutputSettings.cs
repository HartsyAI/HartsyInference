namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Controls where and how generated images are saved. Creates date-based subdirectories with unique filenames.</summary>
public sealed record OutputSettings
{
    /// <summary>Root output directory. Date-based subdirectories are created automatically.</summary>
    public required string OutputDir { get; init; }

    /// <summary>Image format extension (without dot). Default: "bmp".</summary>
    public string Format { get; init; } = "bmp";

    /// <summary>Optional prefix for filenames. Default: empty (uses prompt slug).</summary>
    public string Prefix { get; init; } = "";

    /// <summary>Generates the next unique output path: OutputDir/yyyy-MM-dd/NNN_prefix_timestamp.ext</summary>
    public string GetNextOutputPath(string? promptSlug = null)
    {
        string dateDir = Path.Combine(OutputDir, DateTime.Now.ToString("yyyy-MM-dd"));
        Directory.CreateDirectory(dateDir);

        // Count existing files to get sequence number
        int existingCount = Directory.Exists(dateDir) ? Directory.GetFiles(dateDir, $"*.{Format}").Length : 0;
        int seqNum = existingCount + 1;

        string slug = !string.IsNullOrWhiteSpace(promptSlug)
            ? SanitizeSlug(promptSlug)
            : !string.IsNullOrWhiteSpace(Prefix)
                ? Prefix
                : "img";

        string timestamp = DateTime.Now.ToString("HHmmss");
        string fileName = $"{seqNum:D3}_{slug}_{timestamp}.{Format}";

        return Path.Combine(dateDir, fileName);
    }

    /// <summary>Creates a default OutputSettings pointing to an Output/ directory next to the executable.</summary>
    public static OutputSettings Default => new OutputSettings
    {
        OutputDir = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "Output"),
    };

    private static string SanitizeSlug(string prompt)
    {
        // Take first ~40 chars, replace non-alphanumeric with underscore, trim
        string slug = prompt.Length > 40 ? prompt[..40] : prompt;
        char[] chars = new char[slug.Length];
        for (int i = 0; i < slug.Length; i++)
        {
            char c = slug[i];
            chars[i] = char.IsLetterOrDigit(c) ? c : '_';
        }

        string result = new string(chars).Trim('_');

        // Collapse multiple underscores
        while (result.Contains("__"))
        {
            result = result.Replace("__", "_");
        }

        return result.ToLowerInvariant();
    }
}
