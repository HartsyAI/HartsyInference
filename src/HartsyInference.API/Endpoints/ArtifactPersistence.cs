using HartsyInference.Core.Logging;
using HartsyInference.Engine;

namespace HartsyInference.API.Endpoints;

/// <summary>Writes generated artifacts to disk for the HTTP routes, matching where the CLI already puts them. A failed write must never fail a generation that already succeeded, so every path logs and returns null rather than throwing — the artifact is still in the response body either way.</summary>
public static class ArtifactPersistence
{
    /// <summary>Saves one file; null when the request opted out or the write failed.</summary>
    public static string? Save(NativeArtifactRequest req, byte[] bytes, string slugSource, string extension) =>
        Save(bytes, slugSource, extension, req.Save, req.OutputDir);

    /// <summary>Envelope-free overload for the OpenAI-compat routes, whose request schema has no place to carry <c>save</c>/<c>outputDir</c> — they always save to the output root.</summary>
    public static string? Save(byte[] bytes, string slugSource, string extension, bool? save = null, string? outputDir = null)
    {
        if (save == false || bytes.Length == 0)
            return null;
        try
        {
            return OutputWriter.WriteBytes(bytes, outputDir, slugSource, extension);
        }
        catch (Exception ex)
        {
            Logs.Warning($"[api] generated artifact could not be saved: {ex.Message}");
            return null;
        }
    }

    /// <summary>Saves a multi-file result into one directory; null when the request opted out or the write failed.</summary>
    public static string? SaveGroup(NativeArtifactRequest req, IReadOnlyDictionary<string, byte[]> files, string slugSource, string extension)
    {
        if (req.Save == false || files.Count == 0)
            return null;
        try
        {
            return OutputWriter.WriteGroup(files, req.OutputDir, slugSource, extension);
        }
        catch (Exception ex)
        {
            Logs.Warning($"[api] generated artifacts could not be saved: {ex.Message}");
            return null;
        }
    }
}
