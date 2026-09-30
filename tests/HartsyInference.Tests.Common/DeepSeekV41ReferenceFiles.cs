namespace HartsyInference.Tests.Common;

/// <summary>Locates the committed DeepSeek-V4.1 encoder reference fixtures and the (uncommitted) real tokenizer.json.</summary>
public static class DeepSeekV41ReferenceFiles
{
    /// <summary>Directory holding encoder_reference.json and the upstream golden encoding fixtures.</summary>
    public static string ReferenceDir { get; } = Path.Combine(
        RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "encoder_reference");

    /// <summary>Path of the HF-generated reference dump.</summary>
    public static string ReferenceJson => Path.Combine(ReferenceDir, "encoder_reference.json");

    /// <summary>Path of the upstream golden input/output fixtures directory.</summary>
    public static string EncodingDir => Path.Combine(ReferenceDir, "encoding");

    /// <summary>Real tokenizer.json from DSV41_TOKENIZER_JSON, the reference checkout or the model RAID; null when absent.</summary>
    public static string? FindTokenizerJson()
    {
        string? overridden = Environment.GetEnvironmentVariable("DSV41_TOKENIZER_JSON");
        if (!string.IsNullOrEmpty(overridden)) return File.Exists(overridden) ? overridden : null;
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, "dsv41-ref", "upstream", "tokenizer.json"),
            Path.Combine(TestPaths.ModelsDir, "DeepSeek-V4.1-Flash", "tokenizer.json"),
        ];
        return candidates.FirstOrDefault(File.Exists);
    }
}
