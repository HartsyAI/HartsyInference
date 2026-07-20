using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Fetches a single non-HuggingFace asset (the public-domain CMU pronouncing dictionary) to a local path.
/// Staged through a <c>.tmp</c> file and moved into place, so an interrupted fetch never looks complete.</summary>
internal static class AudioFileFetcher
{
    /// <summary>Downloads <paramref name="url"/> to <paramref name="targetPath"/> unless it is already present.</summary>
    internal static async Task EnsureAsync(string url, string targetPath, CancellationToken cancel)
    {
        if (File.Exists(targetPath))
        {
            return;
        }
        string? directory = Path.GetDirectoryName(targetPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }
        string tempPath = targetPath + ".tmp";
        try
        {
            using HttpClient client = new HttpClient();
            using HttpResponseMessage response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancel).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();
            await using (FileStream file = File.Create(tempPath))
            {
                await response.Content.CopyToAsync(file, cancel).ConfigureAwait(false);
            }
            File.Move(tempPath, targetPath, overwrite: true);
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio] Failed to download '{url}' to '{targetPath}': {ex.Message}", ex);
            TryDeleteTemp(tempPath);
            throw;
        }
    }

    private static void TryDeleteTemp(string tempPath)
    {
        try
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Audio] Could not remove the partial download '{tempPath}': {ex.Message}");
        }
    }
}
