using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Routes LTX-2 dev-family requests whose checkpoint is a distilled build to the distilled sampling
/// contract. The dev and distilled 2.5 transformers are byte-indistinguishable in structure (same model version,
/// config, and tensor keys), so the FILENAME is the only signal there is — the same remap-table approach SwarmUI
/// uses for Hunyuan 1.5 distilled. This is what makes the distilled contract reachable from SwarmUI, which can
/// only send the dev family id.</summary>
public static class LtxVideo2DistilledRouting
{
    /// <summary>The id the distilled <see cref="LtxVideo2Recipe"/> instance is registered under.</summary>
    public const string DistilledFamilyId = "ltx-2.5-distilled";

    /// <summary>Family ids served by the dev <see cref="LtxVideo2Recipe"/> instance.</summary>
    public static bool IsDevFamilyId(string familyId) =>
        string.Equals(familyId, "ltx-video-2", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "ltx-video2", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "lightricks-ltx-video-2", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "ltx-2", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "ltx-2.3", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "ltx-2.5", StringComparison.OrdinalIgnoreCase);

    /// <summary>True when the checkpoint's file name, leaf directory name, or (for a directory) any contained
    /// <c>*.safetensors</c> file name contains "distilled". The directory scan is load-bearing: distilled runs
    /// stage a directory (the distilled file is transformer-only; VAE/Gemma are merged from siblings), so the
    /// leaf name alone would make routing depend on what the user called their folder.</summary>
    public static bool IsDistilledCheckpoint(string? checkpointPath)
    {
        if (string.IsNullOrWhiteSpace(checkpointPath))
            return false;
        if (Path.GetFileName(Path.TrimEndingDirectorySeparator(checkpointPath))
                .Contains("distilled", StringComparison.OrdinalIgnoreCase))
            return true;
        if (!Directory.Exists(checkpointPath))
            return false;
        try
        {
            foreach (string file in Directory.EnumerateFiles(checkpointPath, "*.safetensors", SearchOption.AllDirectories))
            {
                if (Path.GetFileName(file).Contains("distilled", StringComparison.OrdinalIgnoreCase))
                    return true;
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Logs.Error($"Could not scan '{checkpointPath}' for distilled checkpoints; treating as dev.", ex);
        }
        return false;
    }

    private static string? _lastLoggedPath;

    /// <summary>A dev-family LTX-2 id whose checkpoint is a distilled build returns
    /// <see cref="DistilledFamilyId"/>; everything else passes through unchanged. Logs when it fires (once per
    /// distinct path) — the remap changes the sampling contract, and that must be visible in the log.</summary>
    public static string RemapFamilyId(string familyId, string? checkpointPath)
    {
        if (!IsDevFamilyId(familyId) || !IsDistilledCheckpoint(checkpointPath))
            return familyId;
        if (_lastLoggedPath != checkpointPath)
        {
            _lastLoggedPath = checkpointPath;
            Logs.Info($"[LtxVideo2Recipe] Checkpoint '{Path.GetFileName(Path.TrimEndingDirectorySeparator(checkpointPath!))}' "
                + $"under family id '{familyId}' is a distilled build — routing to the {DistilledFamilyId} sampling "
                + "contract (8-step fixed sigmas, cfg 1, two-stage).");
        }
        return DistilledFamilyId;
    }
}
