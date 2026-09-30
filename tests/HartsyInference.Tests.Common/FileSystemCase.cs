namespace HartsyInference.Tests.Common;

/// <summary>Whether a filesystem tells names apart by case, for tests that need two spellings of one name side by
/// side: they cannot be created on Windows or default macOS volumes, so those tests skip there.</summary>
public static class FileSystemCase
{
    /// <summary>True when names in <paramref name="directory"/>, which must exist and be writable, are case-sensitive.</summary>
    public static bool IsCaseSensitive(string directory)
    {
        string name = "case-probe-" + Guid.NewGuid().ToString("N");
        string probe = Path.Combine(directory, name);
        File.WriteAllBytes(probe, []);
        try
        {
            return !File.Exists(Path.Combine(directory, name.ToUpperInvariant()));
        }
        finally
        {
            File.Delete(probe);
        }
    }
}
