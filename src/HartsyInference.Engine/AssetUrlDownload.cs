using System.Security.Cryptography;

namespace HartsyInference.Engine;

/// <summary>Downloads one file from a plain https URL (a weights mirror, a GitHub release asset; a local mirror may use http) with the same staging and hash check the Hugging Face path has: the bytes go to a <c>.tmp</c> file, are hashed against the pinned SHA-256 and only then moved into place.
/// The client it is given must carry no credentials: a Hugging Face token never goes to another host.</summary>
internal static class AssetUrlDownload
{
    private const int BufferSize = 81920;

    /// <summary>The URL a mirror serves <paramref name="asset"/>'s repack at, or null when the asset lists none, <paramref name="baseUrl"/> is empty, not an absolute https URL or carries a query or fragment, or the repack path leaves the mirror's folder.</summary>
    internal static string? MirrorUrl(ModelAsset asset, string? baseUrl)
    {
        if (asset.Repack is null || string.IsNullOrWhiteSpace(baseUrl))
            return null;
        if (!Uri.TryCreate(baseUrl.Trim(), UriKind.Absolute, out Uri? root) || !IsAllowed(root) || root.Query.Length > 0 || root.Fragment.Length > 0)
            return null;
        string[] segments = asset.Repack.MirrorPath.Split('/');
        if (segments.Any(s => s.Length == 0 || s is "." or ".." || s.Contains('\\')))
            return null;
        return root.AbsoluteUri.TrimEnd('/') + "/" + string.Join('/', segments.Select(Uri.EscapeDataString));
    }

    /// <summary>Downloads <paramref name="url"/> to <paramref name="destination"/>, discarding the bytes when they do not hash to <paramref name="sha256"/>.</summary>
    /// <exception cref="ArgumentException">The URL is not an absolute https URL, or no hash is pinned.</exception>
    /// <exception cref="HttpRequestException">The server answered with an error status or the connection failed.</exception>
    /// <exception cref="InvalidOperationException">The bytes did not match the pinned hash.</exception>
    internal static async Task DownloadAsync(HttpClient http, string url, string label, string destination, IProgress<double>? progress, string? sha256, CancellationToken ct)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out Uri? uri) || !IsAllowed(uri))
            throw new ArgumentException($"'{label}' must be fetched from an https URL.", nameof(url));
        if (string.IsNullOrWhiteSpace(sha256))
            throw new ArgumentException($"'{label}' is fetched from a URL and needs a pinned sha256.", nameof(sha256));
        string? directory = Path.GetDirectoryName(destination);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        using HttpResponseMessage response = await http.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        long total = response.Content.Headers.ContentLength ?? -1L;
        string temp = destination + ".tmp";
        try
        {
            using SHA256 hash = SHA256.Create();
            {
                await using Stream source = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
                await using FileStream target = new(temp, FileMode.Create, FileAccess.Write, FileShare.None, BufferSize, useAsync: true);
                byte[] buffer = new byte[BufferSize];
                long received = 0;
                int read;
                while ((read = await source.ReadAsync(buffer, ct).ConfigureAwait(false)) > 0)
                {
                    await target.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
                    hash.TransformBlock(buffer, 0, read, null, 0);
                    received += read;
                    if (total > 0)
                        progress?.Report((double)received / total);
                }
            }
            hash.TransformFinalBlock([], 0, 0);
            string actual = Convert.ToHexString(hash.Hash!).ToLowerInvariant();
            if (!string.Equals(actual, sha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException($"SHA-256 mismatch for \"{label}\": expected {sha256}, got {actual}. The download was discarded.");
            progress?.Report(1.0);
        }
        catch
        {
            TryDelete(temp);
            throw;
        }
        File.Move(temp, destination, overwrite: true);
    }

    /// <summary>https, or plain http to this machine (<c>localhost</c>, <c>127.0.0.1</c>, <c>[::1]</c>) so a local mirror can be tried without a certificate.</summary>
    internal static bool IsAllowed(Uri uri) => uri.Scheme == Uri.UriSchemeHttps || (uri.Scheme == Uri.UriSchemeHttp && uri.IsLoopback);

    private static void TryDelete(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (IOException)
        {
            // The mismatch or network error is what the caller needs; a leftover .tmp is overwritten by the next attempt.
        }
        catch (UnauthorizedAccessException)
        {
            // Same: the original failure is the one to report.
        }
    }
}
