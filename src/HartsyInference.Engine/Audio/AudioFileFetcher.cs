using System.Buffers;
using System.Security.Cryptography;
using HartsyInference.Audio.Cache;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Fetches a single non-HuggingFace asset (the public-domain CMU pronouncing dictionary, Demucs, xiph's RNNoise
/// model) to a local path. Staged through a <c>.tmp</c> file and moved into place, so an interrupted or rejected fetch
/// never looks complete.</summary>
internal static class AudioFileFetcher
{
    /// <summary>Downloads <paramref name="url"/> to <paramref name="targetPath"/> unless it is already present.</summary>
    internal static Task EnsureAsync(string url, string targetPath, CancellationToken cancel) =>
        EnsureAsync(url, targetPath, expectedSha256: null, cancel);

    /// <summary>Downloads <paramref name="url"/> to <paramref name="targetPath"/> unless it is already present.</summary>
    /// <param name="expectedSha256">Hex SHA-256 the bytes must have before they are moved into place; null skips the
    /// check.</param>
    /// <exception cref="InvalidDataException">The download's SHA-256 is not <paramref name="expectedSha256"/>.</exception>
    internal static async Task EnsureAsync(string url, string targetPath, string? expectedSha256,
        CancellationToken cancel)
    {
        using HttpClient client = new HttpClient();
        await EnsureAsync(client, url, targetPath, expectedSha256, cancel).ConfigureAwait(false);
    }

    /// <summary>As <see cref="EnsureAsync(string, string, string?, CancellationToken)"/>, over a caller-supplied
    /// client.</summary>
    internal static async Task EnsureAsync(HttpClient client, string url, string targetPath, string? expectedSha256,
        CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(client);
        if (File.Exists(targetPath))
        {
            return;
        }
        AudioStandIns.Resync(AudioModelRoot.Root());
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
            using HttpResponseMessage response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancel).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();
            string? actual = null;
            await using (FileStream file = File.Create(tempPath))
            {
                if (expectedSha256 is null)
                {
                    await response.Content.CopyToAsync(file, cancel).ConfigureAwait(false);
                }
                else
                {
                    await using Stream body = await response.Content.ReadAsStreamAsync(cancel).ConfigureAwait(false);
                    actual = await CopyHashingAsync(body, file, cancel).ConfigureAwait(false);
                }
            }
            if (expectedSha256 is not null
                && !string.Equals(actual, expectedSha256, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException(
                    $"'{url}' has SHA-256 {actual}, expected {expectedSha256}; the download was discarded.");
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

    /// <summary>Copies <paramref name="source"/> to <paramref name="destination"/> and returns the lowercase hex SHA-256
    /// of what passed through.</summary>
    private static async Task<string> CopyHashingAsync(Stream source, Stream destination, CancellationToken cancel)
    {
        using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        byte[] buffer = ArrayPool<byte>.Shared.Rent(1 << 20);
        try
        {
            int read;
            while ((read = await source.ReadAsync(buffer.AsMemory(), cancel).ConfigureAwait(false)) > 0)
            {
                hash.AppendData(buffer, 0, read);
                await destination.WriteAsync(buffer.AsMemory(0, read), cancel).ConfigureAwait(false);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
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
