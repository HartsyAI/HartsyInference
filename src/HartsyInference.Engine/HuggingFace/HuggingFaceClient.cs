using System.Net.Http.Headers;
using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.HuggingFace;

/// <summary>Describes a model returned by the HuggingFace Hub API.</summary>
public sealed record HuggingFaceModelInfo
{
    /// <summary>Full repository identifier (e.g., "runwayml/stable-diffusion-v1-5").</summary>
    public required string Id { get; init; }

    /// <summary>Author or organization that owns the model.</summary>
    public string? Author { get; init; }

    /// <summary>Short model name without the author prefix.</summary>
    public required string ModelId { get; init; }

    /// <summary>Tags associated with the model on the Hub.</summary>
    public required string[] Tags { get; init; }

    /// <summary>When the model was last modified on the Hub.</summary>
    public DateTime LastModified { get; init; }

    /// <summary>Pipeline tag describing the model's task (e.g., "text-to-image").</summary>
    public string? Pipeline_tag { get; init; }

    /// <summary>Library the model was built with (e.g., "diffusers", "transformers").</summary>
    public string? LibraryName { get; init; }
}

/// <summary>Describes a single file in a HuggingFace model repository.</summary>
public sealed record HuggingFaceFileInfo
{
    /// <summary>Relative file name within the repository.</summary>
    public required string FileName { get; init; }

    /// <summary>File size in bytes.</summary>
    public long Size { get; init; }

    /// <summary>Whether the file is stored with Git LFS.</summary>
    public bool Lfs { get; init; }
}

/// <summary>HTTP client for the HuggingFace Hub API. Supports searching, downloading, and inspecting model repositories.</summary>
public sealed class HuggingFaceClient : IDisposable
{
    private const string BaseUrl = "https://huggingface.co";
    private const string ApiUrl = "https://huggingface.co/api/models";
    private const int DownloadBufferSize = 81920;

    private readonly HttpClient _httpClient;
    private readonly bool _ownsHttpClient;
    private readonly string? _token;
    private int _disposed;

    /// <summary>Creates a new client, optionally authenticating with a HuggingFace token. Falls back to the HF_TOKEN environment variable.</summary>
    public HuggingFaceClient(string? token = null, HttpClient? httpClient = null)
    {
        _token = token ?? Environment.GetEnvironmentVariable("HF_TOKEN");
        _ownsHttpClient = httpClient is null;
        _httpClient = httpClient ?? new HttpClient();

        if (_token is not null)
        {
            _httpClient.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", _token);
        }

        _httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("HartsyInference/1.0");
    }

    /// <summary>Searches the HuggingFace Hub for models matching the given query and optional tag.</summary>
    public async Task<IReadOnlyList<HuggingFaceModelInfo>> SearchModelsAsync(
        string query,
        string? tag,
        int limit,
        CancellationToken ct)
    {
        ThrowIfDisposed();

        string url = $"{ApiUrl}?search={Uri.EscapeDataString(query)}&limit={limit}";
        if (tag is not null)
        {
            url += $"&filter={Uri.EscapeDataString(tag)}";
        }

        Logs.Debug($"HuggingFace: searching models query=\"{query}\" tag=\"{tag}\" limit={limit}");

        HttpResponseMessage response = await _httpClient.GetAsync(url, ct).ConfigureAwait(false);
        await EnsureSuccessAsync(response, $"search models (query=\"{query}\")").ConfigureAwait(false);

        byte[] body = await response.Content.ReadAsByteArrayAsync(ct).ConfigureAwait(false);
        using JsonDocument doc = JsonDocument.Parse(body);
        JsonElement root = doc.RootElement;

        List<HuggingFaceModelInfo> results = new List<HuggingFaceModelInfo>();
        foreach (JsonElement element in root.EnumerateArray())
        {
            HuggingFaceModelInfo info = ParseModelInfo(element);
            results.Add(info);
        }

        Logs.Info($"HuggingFace: search returned {results.Count} models for query=\"{query}\"");
        return results;
    }

    /// <summary>Retrieves detailed information about a single model repository.</summary>
    public async Task<HuggingFaceModelInfo> GetModelInfoAsync(string repoId, CancellationToken ct)
    {
        ThrowIfDisposed();

        string url = $"{ApiUrl}/{repoId}";
        Logs.Debug($"HuggingFace: fetching model info for \"{repoId}\"");

        HttpResponseMessage response = await _httpClient.GetAsync(url, ct).ConfigureAwait(false);
        await EnsureSuccessAsync(response, $"get model info for \"{repoId}\"").ConfigureAwait(false);

        byte[] body = await response.Content.ReadAsByteArrayAsync(ct).ConfigureAwait(false);
        using JsonDocument doc = JsonDocument.Parse(body);
        JsonElement root = doc.RootElement;

        HuggingFaceModelInfo info = ParseModelInfo(root);
        Logs.Info($"HuggingFace: retrieved info for \"{repoId}\"");
        return info;
    }

    /// <summary>Downloads a single file from a model repository to the specified local path, reporting progress as a
    /// fraction from 0.0 to 1.0. Stages to a <c>.tmp</c> file, optionally verifies its SHA-256 against
    /// <paramref name="sha256"/> (a corrupt mid-flight download is deleted and the call fails), then atomically moves
    /// it into place.</summary>
    public async Task DownloadFileAsync(
        string repoId,
        string fileName,
        string destinationPath,
        IProgress<double>? progress,
        string? sha256,
        CancellationToken ct)
    {
        ThrowIfDisposed();

        string url = $"{BaseUrl}/{repoId}/resolve/main/{fileName}";
        Logs.Info($"HuggingFace: downloading \"{repoId}/{fileName}\" -> \"{destinationPath}\"");

        using HttpResponseMessage response = await _httpClient
            .GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct)
            .ConfigureAwait(false);
        await EnsureSuccessAsync(response, $"download \"{repoId}/{fileName}\"").ConfigureAwait(false);

        long totalBytes = response.Content.Headers.ContentLength ?? -1L;
        string? directory = Path.GetDirectoryName(destinationPath);
        if (directory is not null)
        {
            Directory.CreateDirectory(directory);
        }

        string tempPath = destinationPath + ".tmp";
        try
        {
            // The write stream must be fully disposed (releasing its FileShare.None lock) before
            // VerifySha256Async reopens tempPath for reading below — nested block so the `await using`
            // cleanup runs at THIS scope's closing brace, not the outer try's, otherwise the verify-read
            // races the still-open write handle and throws "process cannot access the file".
            {
                await using Stream sourceStream = await response.Content
                    .ReadAsStreamAsync(ct)
                    .ConfigureAwait(false);
                await using FileStream destinationStream = new FileStream(
                    tempPath,
                    FileMode.Create,
                    FileAccess.Write,
                    FileShare.None,
                    DownloadBufferSize,
                    useAsync: true);

                byte[] buffer = new byte[DownloadBufferSize];
                long bytesRead = 0L;
                int read;

                while ((read = await sourceStream.ReadAsync(buffer, ct).ConfigureAwait(false)) > 0)
                {
                    await destinationStream.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
                    bytesRead += read;

                    if (totalBytes > 0)
                    {
                        double fraction = (double)bytesRead / totalBytes;
                        progress?.Report(fraction);
                    }
                }
            }

            progress?.Report(1.0);

            if (!string.IsNullOrEmpty(sha256))
                await VerifySha256Async(tempPath, sha256, $"{repoId}/{fileName}", ct).ConfigureAwait(false);
        }
        catch
        {
            TryDeleteFile(tempPath);
            throw;
        }

        File.Move(tempPath, destinationPath, overwrite: true);
        Logs.Info($"HuggingFace: download complete \"{repoId}/{fileName}\" ({new FileInfo(destinationPath).Length} bytes)");
    }

    /// <summary>Computes the SHA-256 of <paramref name="path"/> and throws when it does not match <paramref name="expected"/>.</summary>
    private static async Task VerifySha256Async(string path, string expected, string label, CancellationToken ct)
    {
        await using FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, DownloadBufferSize, useAsync: true);
        byte[] hash = await System.Security.Cryptography.SHA256.HashDataAsync(stream, ct).ConfigureAwait(false);
        string actual = Convert.ToHexString(hash).ToLowerInvariant();
        if (!string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                $"SHA-256 mismatch for \"{label}\": expected {expected}, got {actual}. The download was corrupted; the partial file was discarded.");
        }
    }

    /// <summary>Lists all files in a model repository.</summary>
    public async Task<IReadOnlyList<HuggingFaceFileInfo>> ListModelFilesAsync(string repoId, CancellationToken ct)
    {
        ThrowIfDisposed();

        string url = $"{ApiUrl}/{repoId}";
        Logs.Debug($"HuggingFace: listing files for \"{repoId}\"");

        HttpResponseMessage response = await _httpClient.GetAsync(url, ct).ConfigureAwait(false);
        await EnsureSuccessAsync(response, $"list files for \"{repoId}\"").ConfigureAwait(false);

        byte[] body = await response.Content.ReadAsByteArrayAsync(ct).ConfigureAwait(false);
        using JsonDocument doc = JsonDocument.Parse(body);
        JsonElement root = doc.RootElement;

        List<HuggingFaceFileInfo> files = new List<HuggingFaceFileInfo>();

        if (root.TryGetProperty("siblings", out JsonElement siblings))
        {
            foreach (JsonElement sibling in siblings.EnumerateArray())
            {
                string rfilename = sibling.GetProperty("rfilename").GetString()!;
                long size = 0;
                bool lfs = false;

                if (sibling.TryGetProperty("size", out JsonElement sizeElement))
                {
                    size = sizeElement.GetInt64();
                }

                if (sibling.TryGetProperty("lfs", out JsonElement lfsElement))
                {
                    lfs = lfsElement.ValueKind != JsonValueKind.Null;
                }

                files.Add(new HuggingFaceFileInfo
                {
                    FileName = rfilename,
                    Size = size,
                    Lfs = lfs,
                });
            }
        }

        Logs.Info($"HuggingFace: found {files.Count} files in \"{repoId}\"");
        return files;
    }

    /// <summary>Resolves the preferred model variant for download: fp16 safetensors, then fp32 safetensors, then GGUF.</summary>
    public async Task<string?> ResolveModelVariantAsync(string repoId, CancellationToken ct)
    {
        ThrowIfDisposed();

        IReadOnlyList<HuggingFaceFileInfo> files = await ListModelFilesAsync(repoId, ct).ConfigureAwait(false);

        Logs.Debug($"HuggingFace: resolving preferred variant for \"{repoId}\" ({files.Count} files)");

        bool hasFp16SafeTensors = false;
        bool hasFp32SafeTensors = false;
        bool hasGguf = false;

        foreach (HuggingFaceFileInfo file in files)
        {
            string name = file.FileName;

            if (name.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            {
                if (name.Contains("fp16", StringComparison.OrdinalIgnoreCase))
                {
                    hasFp16SafeTensors = true;
                }
                else
                {
                    hasFp32SafeTensors = true;
                }
            }
            else if (name.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            {
                hasGguf = true;
            }
        }

        string? variant;
        if (hasFp16SafeTensors)
        {
            variant = "fp16-safetensors";
        }
        else if (hasFp32SafeTensors)
        {
            variant = "fp32-safetensors";
        }
        else if (hasGguf)
        {
            variant = "gguf";
        }
        else
        {
            variant = null;
        }

        Logs.Info($"HuggingFace: resolved variant for \"{repoId}\": {variant ?? "(none)"}");
        return variant;
    }

    /// <summary>Releases the underlying HTTP client if this instance owns it.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        if (_ownsHttpClient)
        {
            _httpClient.Dispose();
        }
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
        {
            throw new ObjectDisposedException(nameof(HuggingFaceClient));
        }
    }

    private static HuggingFaceModelInfo ParseModelInfo(JsonElement element)
    {
        string id = element.GetProperty("id").GetString()!;
        string modelId = element.TryGetProperty("modelId", out JsonElement modelIdElement)
            ? modelIdElement.GetString() ?? id
            : id;
        string? author = element.TryGetProperty("author", out JsonElement authorElement)
            ? authorElement.GetString()
            : null;

        List<string> tags = new List<string>();
        if (element.TryGetProperty("tags", out JsonElement tagsElement))
        {
            foreach (JsonElement tag in tagsElement.EnumerateArray())
            {
                string? tagValue = tag.GetString();
                if (tagValue is not null)
                {
                    tags.Add(tagValue);
                }
            }
        }

        DateTime lastModified = DateTime.UtcNow;
        if (element.TryGetProperty("lastModified", out JsonElement lastModElement))
        {
            string? dateStr = lastModElement.GetString();
            if (dateStr is not null && DateTime.TryParse(dateStr, out DateTime parsed))
            {
                lastModified = parsed;
            }
        }

        string? pipelineTag = element.TryGetProperty("pipeline_tag", out JsonElement ptElement)
            ? ptElement.GetString()
            : null;

        string? libraryName = element.TryGetProperty("library_name", out JsonElement lnElement)
            ? lnElement.GetString()
            : null;

        return new HuggingFaceModelInfo
        {
            Id = id,
            Author = author,
            ModelId = modelId,
            Tags = tags.ToArray(),
            LastModified = lastModified,
            Pipeline_tag = pipelineTag,
            LibraryName = libraryName,
        };
    }

    private static async Task EnsureSuccessAsync(HttpResponseMessage response, string operation)
    {
        if (!response.IsSuccessStatusCode)
        {
            string body = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            throw new HartsyInferenceException(
                $"HuggingFace API request failed for {operation}: HTTP {(int)response.StatusCode} {response.ReasonPhrase}. Body: {body}");
        }
    }

    private static void TryDeleteFile(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (Exception ex)
        {
            Logs.Verbose($"Best-effort delete of {path} failed: {ex.Message}");
        }
    }
}
