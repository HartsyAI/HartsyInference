using System.Net;
using System.Net.Http.Headers;

namespace HartsyInference.Audio.Cache;

/// <summary>Resolves HuggingFace model artifacts to a local file path, downloading them
/// on first use into a shared on-disk cache at <c>~/.cache/hartsyinference/models/</c>.
///
/// <para>The cache layout mirrors HuggingFace conventions so a given repo lives under
/// <c>~/.cache/hartsyinference/models/{owner}--{name}/</c>. Multi-file repos (Whisper has
/// <c>model.safetensors</c>, <c>config.json</c>, <c>tokenizer.json</c>, etc.) all land in
/// the same directory.</para>
///
/// <para>Override the cache root with the <c>HARTSYINFERENCE_MODEL_CACHE</c> environment
/// variable. Override the HuggingFace mirror with <c>HF_ENDPOINT</c>. Pass a token in
/// <c>HF_TOKEN</c> for gated repos.</para>
///
/// <para><b>Thread safety:</b> safe to call <see cref="GetAsync"/> concurrently from
/// multiple threads — concurrent downloads of the same file are coalesced via a per-path
/// async lock so we don't re-download or corrupt the file. The first caller does the
/// download; subsequent callers await the same task.</para></summary>
public static class AudioModelCache
{
    private const string DefaultEndpoint = "https://huggingface.co";

    private static readonly Lazy<HttpClient> _http = new(CreateHttpClient);
    private static readonly Dictionary<string, Task<string>> _inflight = new(StringComparer.Ordinal);
    private static readonly object _inflightLock = new();

    /// <summary>The active cache root. Honors <c>HARTSYINFERENCE_MODEL_CACHE</c>;
    /// defaults to <c>~/.cache/hartsyinference/models</c>.</summary>
    public static string CacheRoot { get; } = ResolveCacheRoot();

    /// <summary>Returns the local directory that holds files for <paramref name="hfRepoId"/>
    /// (e.g. <c>"openai/whisper-large-v3-turbo"</c>). The directory is created if it
    /// does not already exist; files inside it may or may not be present yet.</summary>
    public static string GetRepoDirectory(string hfRepoId)
    {
        // The double-dash convention matches HF's own hub cache layout, so a user who
        // already has whisper downloaded under ~/.cache/huggingface/hub can symlink it
        // into ours without renaming. Only the org/name separator becomes `--`; existing
        // dashes in repo names stay single.
        string safe = hfRepoId.Replace("/", "--", StringComparison.Ordinal);
        string dir = Path.Combine(CacheRoot, safe);
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>Resolves a single file to a local path, downloading it from HuggingFace
    /// if not already cached. Returns the absolute path to the file.</summary>
    /// <param name="hfRepoId">Repo id such as <c>"openai/whisper-large-v3-turbo"</c>.</param>
    /// <param name="filename">File path within the repo, e.g. <c>"model.safetensors"</c>
    /// or <c>"tokenizer.json"</c>. Subdirectories are allowed.</param>
    /// <param name="revision">Optional git revision (branch / tag / commit SHA).
    /// Defaults to <c>"main"</c>.</param>
    /// <param name="progress">Optional progress callback receiving downloaded-byte counts.</param>
    public static async Task<string> GetAsync(
        string hfRepoId,
        string filename,
        string revision = "main",
        IProgress<long>? progress = null,
        CancellationToken ct = default)
    {
        string repoDir = GetRepoDirectory(hfRepoId);
        string localPath = Path.Combine(repoDir, filename);
        if (File.Exists(localPath)) return localPath;

        // Coalesce concurrent requests for the same file so we don't double-download.
        Task<string> downloadTask;
        lock (_inflightLock)
        {
            if (!_inflight.TryGetValue(localPath, out Task<string>? existing))
            {
                downloadTask = DownloadAsync(hfRepoId, filename, revision, localPath, progress, ct);
                _inflight[localPath] = downloadTask;
            }
            else
            {
                downloadTask = existing;
            }
        }

        try
        {
            return await downloadTask.ConfigureAwait(false);
        }
        finally
        {
            lock (_inflightLock) _inflight.Remove(localPath);
        }
    }

    /// <summary>Synchronous wrapper around <see cref="GetAsync"/>. Convenience for code
    /// that loads at startup time and doesn't want to plumb async through.</summary>
    public static string Get(string hfRepoId, string filename, string revision = "main")
        => GetAsync(hfRepoId, filename, revision).GetAwaiter().GetResult();

    private static async Task<string> DownloadAsync(
        string hfRepoId,
        string filename,
        string revision,
        string localPath,
        IProgress<long>? progress,
        CancellationToken ct)
    {
        string endpoint = Environment.GetEnvironmentVariable("HF_ENDPOINT") ?? DefaultEndpoint;
        string url = $"{endpoint}/{hfRepoId}/resolve/{revision}/{filename}";

        Directory.CreateDirectory(Path.GetDirectoryName(localPath)!);

        // Stream into a temp file in the same directory so the final rename is atomic
        // even on a system that crashes mid-download. Half-finished `.partial` files
        // are obviously truncated and safe to delete on retry.
        string tempPath = localPath + ".partial";

        using HttpRequestMessage request = new(HttpMethod.Get, url);
        using HttpResponseMessage response = await _http.Value
            .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct)
            .ConfigureAwait(false);

        if (response.StatusCode == HttpStatusCode.NotFound)
        {
            throw new FileNotFoundException(
                $"HuggingFace file not found: {hfRepoId}/{filename} @ {revision}. " +
                $"Check the repo id and filename, or set HF_TOKEN if the repo is gated.",
                url);
        }
        if (response.StatusCode == HttpStatusCode.Unauthorized || response.StatusCode == HttpStatusCode.Forbidden)
        {
            throw new UnauthorizedAccessException(
                $"HuggingFace returned {(int)response.StatusCode} for {url}. " +
                "If this is a gated repo, set the HF_TOKEN environment variable to a token " +
                "with read access. Visit https://huggingface.co/settings/tokens to create one.");
        }
        response.EnsureSuccessStatusCode();

        long? total = response.Content.Headers.ContentLength;
        long downloaded = 0;

        await using (Stream src = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false))
        await using (FileStream dst = new(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 1 << 16, useAsync: true))
        {
            byte[] buffer = new byte[1 << 16];
            int read;
            while ((read = await src.ReadAsync(buffer.AsMemory(), ct).ConfigureAwait(false)) > 0)
            {
                await dst.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
                downloaded += read;
                progress?.Report(downloaded);
            }
        }

        File.Move(tempPath, localPath, overwrite: true);
        _ = total; // suppress unused-warning; useful for future progress%
        return localPath;
    }

    private static string ResolveCacheRoot()
    {
        string? env = Environment.GetEnvironmentVariable("HARTSYINFERENCE_MODEL_CACHE");
        if (!string.IsNullOrEmpty(env)) return env;
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(home, ".cache", "hartsyinference", "models");
    }

    private static HttpClient CreateHttpClient()
    {
        HttpClient client = new(new SocketsHttpHandler
        {
            AutomaticDecompression = DecompressionMethods.All,
            // HF redirects to CloudFront / LFS; allow the client to follow them.
            AllowAutoRedirect = true,
            MaxConnectionsPerServer = 8,
        })
        {
            // Some safetensors weight files are multi-GB and slow on cold links.
            Timeout = TimeSpan.FromMinutes(60),
        };
        client.DefaultRequestHeaders.UserAgent.Add(
            new ProductInfoHeaderValue("HartsyInference.Audio", "0.1"));

        string? token = Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);

        return client;
    }
}
