using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Cache;

/// <summary>Resolves HuggingFace model artifacts to a local file path, downloading them
/// on first use into a shared on-disk cache under the audio Models tree, organized by category
/// (<c>tts</c>/<c>stt</c>/<c>music</c>/<c>clone</c>/<c>fx</c>/<c>misc</c>) — the same taxonomy
/// <c>AudioModelRoot.WeightsDirectory</c> (Engine) uses for placed-checkpoint providers, so both
/// download paths land under one tree instead of splintering into a separate flat cache.
///
/// <para>The cache layout mirrors HuggingFace conventions so a given repo lives under
/// <c>{root}/{category}/{owner}--{name}/</c>. Multi-file repos (Whisper has
/// <c>model.safetensors</c>, <c>config.json</c>, <c>tokenizer.json</c>, etc.) all land in
/// the same directory.</para>
///
/// <para>Root resolution, highest priority first: the <c>HARTSYINFERENCE_MODEL_CACHE</c>
/// environment variable (an explicit override); else <c>{HARTSYINFERENCE_MODELS}/audio</c> when
/// that's set (aligning with every other model path in the engine); else
/// <c>~/.cache/hartsyinference/models</c> (standalone fallback, e.g. running outside the Engine's
/// placement). Override the HuggingFace mirror with <c>HF_ENDPOINT</c>. Pass a token in
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

    /// <summary>The active cache root (category subfolders live underneath this — see
    /// <see cref="GetRepoDirectory"/>). See the class doc for the resolution order.</summary>
    public static string CacheRoot { get; } = ResolveCacheRoot();

    /// <summary>Returns the local directory that holds files for <paramref name="hfRepoId"/>
    /// (e.g. <c>"openai/whisper-large-v3-turbo"</c>) under <paramref name="category"/>
    /// (<c>"tts"</c>/<c>"stt"</c>/<c>"music"</c>/<c>"clone"</c>/<c>"fx"</c>/<c>"misc"</c>). The
    /// directory is created if it does not already exist; files inside it may or may not be
    /// present yet.</summary>
    public static string GetRepoDirectory(string hfRepoId, string category)
    {
        // The double-dash convention matches HF's own hub cache layout, so a user who
        // already has whisper downloaded under ~/.cache/huggingface/hub can symlink it
        // into ours without renaming. Only the org/name separator becomes `--`; existing
        // dashes in repo names stay single.
        string safe = hfRepoId.Replace("/", "--", StringComparison.Ordinal);
        string dir = Path.Combine(CacheRoot, category, safe);
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>Resolves a single file to a local path, downloading it from HuggingFace
    /// if not already cached. Returns the absolute path to the file.</summary>
    /// <param name="hfRepoId">Repo id such as <c>"openai/whisper-large-v3-turbo"</c>.</param>
    /// <param name="filename">File path within the repo, e.g. <c>"model.safetensors"</c>
    /// or <c>"tokenizer.json"</c>. Subdirectories are allowed.</param>
    /// <param name="category">The audio category this model belongs to — see <see cref="GetRepoDirectory"/>.</param>
    /// <param name="revision">Optional git revision (branch / tag / commit SHA).
    /// Defaults to <c>"main"</c>.</param>
    /// <param name="progress">Optional progress callback receiving downloaded-byte counts.</param>
    public static async Task<string> GetAsync(
        string hfRepoId,
        string filename,
        string category,
        string revision = "main",
        IProgress<long>? progress = null,
        CancellationToken ct = default)
    {
        string repoDir = GetRepoDirectory(hfRepoId, category);
        string localPath = Path.Combine(repoDir, filename);
        if (IsUsableFile(localPath)) return localPath;

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

    /// <summary>Resolves every file in <paramref name="files"/>, downloading whatever is missing, and returns
    /// them keyed by <see cref="AudioModelFile.Name"/>.
    ///
    /// <para>A pipeline's load path and its prefetch both go through this with the same list, which is what
    /// keeps "installed" honest: a model cannot report a clean install and then fail to load on a file the
    /// installer never fetched. Optional entries that the repo does not have are simply absent from the result.</para>
    ///
    /// <para>The required file listed LAST is fetched last on purpose. Presence of the primary artifact is what
    /// the model scanner treats as "this model is installed", so it must not appear until its companions are
    /// already on disk — otherwise an interrupted download leaves a model that looks selectable and cannot load.</para></summary>
    /// <param name="progress">Reports each file as it completes, for install UIs.</param>
    public static async Task<IReadOnlyDictionary<string, string>> FetchAllAsync(
        string hfRepoId,
        IReadOnlyList<AudioModelFile> files,
        string category,
        string revision = "main",
        IProgress<AudioFetchProgress>? progress = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(files);
        Dictionary<string, string> resolved = new(StringComparer.Ordinal);
        // Everything but the last required file downloads concurrently; the last one is awaited afterwards so
        // it cannot land before its companions.
        int lastRequired = -1;
        for (int i = 0; i < files.Count; i++)
        {
            if (files[i].Required)
            {
                lastRequired = i;
            }
        }
        List<(AudioModelFile File, Task<string?> Task)> pending = [];
        for (int i = 0; i < files.Count; i++)
        {
            if (i == lastRequired)
            {
                continue;
            }
            pending.Add((files[i], FetchOneAsync(files[i].Repo ?? hfRepoId, files[i], category, revision, ct)));
        }
        int done = 0;
        int total = files.Count;
        foreach ((AudioModelFile file, Task<string?> task) in pending)
        {
            string? path = await task.ConfigureAwait(false);
            done++;
            if (path is not null)
            {
                resolved[file.Name] = path;
                progress?.Report(new AudioFetchProgress(file.Name, done, total));
            }
        }
        if (lastRequired >= 0)
        {
            AudioModelFile primary = files[lastRequired];
            string? path = await FetchOneAsync(primary.Repo ?? hfRepoId, primary, category, revision, ct).ConfigureAwait(false);
            if (path is not null)
            {
                resolved[primary.Name] = path;
            }
            progress?.Report(new AudioFetchProgress(primary.Name, total, total));
        }
        return resolved;
    }

    /// <summary>Whether the repo has this file, without downloading it — a cached copy answers immediately,
    /// otherwise a HEAD request does. Lets a caller discover a checkpoint's layout (single-file vs sharded)
    /// without pulling gigabytes just to find out which one it is.</summary>
    public static async Task<bool> ExistsAsync(string hfRepoId, string filename, string category, string revision = "main",
        CancellationToken ct = default)
    {
        string localPath = Path.Combine(GetRepoDirectory(hfRepoId, category), filename);
        if (IsUsableFile(localPath))
        {
            return true;
        }
        string endpoint = Environment.GetEnvironmentVariable("HF_ENDPOINT") ?? DefaultEndpoint;
        string url = $"{endpoint}/{hfRepoId}/resolve/{revision}/{filename}";
        try
        {
            using HttpRequestMessage request = new(HttpMethod.Head, url);
            using HttpResponseMessage response = await _http.Value.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct)
                .ConfigureAwait(false);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            Logs.Debug($"[AudioCache] Could not probe '{hfRepoId}/{filename}': {ex.Message}");
            return false;
        }
    }

    private static async Task<string?> FetchOneAsync(string hfRepoId, AudioModelFile file, string category, string revision,
        CancellationToken ct)
    {
        try
        {
            string path = await GetAsync(hfRepoId, file.Name, category, revision, progress: null, ct).ConfigureAwait(false);
            if (file.Sha256 is not null)
            {
                VerifySha256(path, file.Sha256);
            }
            return path;
        }
        catch (Exception ex) when (!file.Required && ex is not OperationCanceledException)
        {
            Logs.Debug($"[AudioCache] Optional file '{file.Name}' not available in '{hfRepoId}': {ex.Message}");
            return null;
        }
    }

    /// <summary>Throws if <paramref name="filePath"/>'s SHA-256 does not match the pinned
    /// <paramref name="expectedHex"/> (case-insensitive). Used to verify a repacked single-file
    /// model against the hash recorded in its repack manifest.</summary>
    public static void VerifySha256(string filePath, string expectedHex)
    {
        using FileStream fs = new(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 20, FileOptions.SequentialScan);
        byte[] hash = SHA256.HashData(fs);
        string actual = Convert.ToHexString(hash);
        if (!actual.Equals(expectedHex, StringComparison.OrdinalIgnoreCase))
        {
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                $"SHA-256 mismatch for '{filePath}': expected {expectedHex}, got {actual}. " +
                "The cached file may be corrupt or stale; delete it to force a re-download.");
        }
    }

    /// <summary>True when the cached entry is a real, openable file. Cache entries are often SYMLINKS into
    /// <c>~/.cache/huggingface/hub</c> (so an existing hub download isn't duplicated), and on .NET
    /// <see cref="File.Exists"/> reports the LINK's existence, not the target's — after a hub cache cleanup a
    /// dangling link would be returned as "cached" and the loader dies with FileNotFound (hit by the 2026-07-24
    /// audio sweep: every Zonos weight was a dangling link). A dangling link is treated as missing so the
    /// download self-heals over it.</summary>
    private static bool IsUsableFile(string path)
    {
        FileInfo info = new(path);
        if (!info.Exists) return false;
        if (info.LinkTarget is null) return true;
        FileSystemInfo? target = info.ResolveLinkTarget(returnFinalTarget: true);
        return target is not null && target.Exists;
    }

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
        // A dangling symlink at the destination must go before the atomic move-over (File.Move would
        // replace the link file itself, but deleting up front also covers the .partial rename path).
        if (File.Exists(localPath) && !IsUsableFile(localPath)) File.Delete(localPath);

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

        // Refuse a download that cannot fit rather than filling the disk and taking the whole process
        // (and every other write on the box) down with it.
        EnsureRoomFor(localPath, total);

        string sizeText = total.HasValue ? $"{total.Value / 1048576.0:0.#} MB" : "unknown size";
        Logs.Info($"[AudioCache] Downloading {hfRepoId}/{filename} ({sizeText})...");
        long nextLogBytes = ProgressLogInterval;
        DateTime started = DateTime.UtcNow;
        try
        {
            // The client's overall timeout has to stay generous for multi-GB files, which means a CONNECTION
            // THAT SIMPLY STOPS DELIVERING looks identical to a slow download for an hour. Bound the gap
            // between reads instead, and reset it whenever bytes actually arrive.
            using CancellationTokenSource stallCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            stallCts.CancelAfter(StallTimeout);
            await using (Stream src = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false))
            await using (FileStream dst = new(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 1 << 16, useAsync: true))
            {
                byte[] buffer = new byte[1 << 16];
                int read;
                while ((read = await ReadWithStallGuard(src, buffer, stallCts, ct, hfRepoId, filename).ConfigureAwait(false)) > 0)
                {
                    stallCts.CancelAfter(StallTimeout);
                    await dst.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
                    downloaded += read;
                    progress?.Report(downloaded);
                    // Without this a multi-GB first load looks like a frozen UI for minutes on end.
                    if (downloaded >= nextLogBytes)
                    {
                        nextLogBytes = downloaded + ProgressLogInterval;
                        double mb = downloaded / 1048576.0;
                        double secs = Math.Max(0.001, (DateTime.UtcNow - started).TotalSeconds);
                        string pct = total is > 0 ? $"{downloaded * 100.0 / total.Value:0.0}% of {total.Value / 1048576.0:0.#} MB" : $"{mb:0.#} MB";
                        Logs.Info($"[AudioCache] {filename}: {pct} ({mb / secs:0.#} MB/s)");
                    }
                }
            }
        }
        catch
        {
            // A half-written .partial is dead weight that survives the failure and eats the disk until
            // someone notices — the exact shape that filled the root filesystem on 2026-08-08.
            TryDelete(tempPath);
            throw;
        }

        File.Move(tempPath, localPath, overwrite: true);
        Logs.Info($"[AudioCache] {filename} ready ({downloaded / 1048576.0:0.#} MB).");
        return localPath;
    }

    /// <summary>How long the transfer may go without receiving a single byte before it is declared stalled.</summary>
    private static readonly TimeSpan StallTimeout = TimeSpan.FromSeconds(120);

    /// <summary>One read, translating a stall-timeout cancellation into a clear error. A caller-requested
    /// cancellation still propagates as cancellation.</summary>
    private static async Task<int> ReadWithStallGuard(Stream src, byte[] buffer, CancellationTokenSource stallCts,
        CancellationToken callerToken, string hfRepoId, string filename)
    {
        try
        {
            return await src.ReadAsync(buffer.AsMemory(), stallCts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!callerToken.IsCancellationRequested)
        {
            throw new TimeoutException(
                $"Download of {hfRepoId}/{filename} stalled — no data for {StallTimeout.TotalSeconds:0} seconds. " +
                "Check the network connection and retry; the partial file has been removed.");
        }
    }

    /// <summary>How much must download between progress log lines.</summary>
    private const long ProgressLogInterval = 128L * 1024 * 1024;

    /// <summary>Free bytes to leave unused after a download, so filling the cache never wedges the machine.</summary>
    private const long DiskHeadroomBytes = 2L * 1024 * 1024 * 1024;

    /// <summary>Throws when <paramref name="needed"/> would not fit on the target volume with headroom to spare.
    /// A clear up-front failure beats an ENOSPC part-way through a multi-GB transfer, which leaves a large
    /// .partial behind and can take the rest of the system down with it.</summary>
    private static void EnsureRoomFor(string localPath, long? needed)
    {
        if (needed is not > 0)
        {
            return;
        }
        try
        {
            string? root = Path.GetPathRoot(Path.GetFullPath(localPath));
            if (string.IsNullOrEmpty(root))
            {
                return;
            }
            long free = new DriveInfo(root).AvailableFreeSpace;
            if (free < needed.Value + DiskHeadroomBytes)
            {
                throw new IOException(
                    $"Not enough disk space: {needed.Value / 1073741824.0:0.##} GB needed (plus " +
                    $"{DiskHeadroomBytes / 1073741824.0:0.#} GB headroom) but only {free / 1073741824.0:0.##} GB free on '{root}'. " +
                    "Free up space or move the model cache with HARTSYINFERENCE_MODEL_CACHE.");
            }
        }
        catch (Exception ex) when (ex is not IOException)
        {
            // Can't determine free space (unusual mount, permissions) — don't block the download over it.
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
            // Best-effort cleanup; the original failure is what matters.
        }
    }

    private static string ResolveCacheRoot()
    {
        string? cacheOverride = Environment.GetEnvironmentVariable("HARTSYINFERENCE_MODEL_CACHE");
        if (!string.IsNullOrEmpty(cacheOverride)) return cacheOverride;
        // Read via the env var (not a reference to RepoPaths/AudioModelRoot) to respect the
        // Audio package's dependency direction -- Engine depends on Audio, not the reverse.
        string? modelsRoot = Environment.GetEnvironmentVariable("HARTSYINFERENCE_MODELS");
        if (!string.IsNullOrEmpty(modelsRoot)) return Path.Combine(modelsRoot, "audio");
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
