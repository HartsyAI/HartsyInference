using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using HartsyInference.Core.Logging;

namespace HartsyInference.Cuda;

/// <summary>One NVIDIA redist archive to provision: where to get it, how to check it, and which libraries to keep from it.</summary>
/// <param name="Component">Cache folder and log tag, e.g. <c>cublas</c> or <c>cudnn</c>.</param>
/// <param name="Label">Name used in log lines, e.g. <c>cuBLAS</c>.</param>
/// <param name="Url">An http(s) URL, or a local archive path (plain or <c>file://</c>) for air-gapped installs and tests.</param>
/// <param name="CudaMajor">The CUDA major the archive is built for; selects the cache folder.</param>
/// <param name="ExpectedSha256">Lower-case hex sha256 of the archive, or null to skip the check.</param>
/// <param name="ExpectedBytes">Archive size for progress lines when the server sends no length, or null.</param>
/// <param name="LibraryPatterns">File patterns (searched recursively in the extracted archive) copied flat into the lib dir.</param>
/// <param name="Sentinel">The file whose presence in the lib dir means the fetch worked.</param>
/// <param name="ApproxSize">Human size for the one-time notice, e.g. <c>about 0.4 GB</c>.</param>
/// <param name="KnobHint">The knob that turns the fetch off, named in the notice and in failures.</param>
public sealed record CudaRedistSpec(
    string Component,
    string Label,
    string Url,
    int CudaMajor,
    string? ExpectedSha256,
    long? ExpectedBytes,
    IReadOnlyList<string> LibraryPatterns,
    string Sentinel,
    string ApproxSize,
    string KnobHint);

/// <summary>Downloads, verifies and unpacks NVIDIA redist archives into the per-user engine cache. Shared by
/// <see cref="CudnnRuntime"/> and <see cref="CublasRuntime"/>; each runtime owns its own probe, guard and trigger points.</summary>
/// <remarks>Layout: <c>~/.cache/hartsyinference/&lt;component&gt;/cuda&lt;N&gt;/lib</c>, one folder per component so a partial
/// extract of one library can never shadow a good copy of another.</remarks>
public static class CudaRedistProvisioner
{
    private const int CopyChunk = 1 << 20;

    /// <summary>Test hook: replaces <c>~/.cache/hartsyinference</c> so tests never touch the real cache.</summary>
    internal static string? CacheRootOverride { get; set; }

    /// <summary>The engine's per-user cache root.</summary>
    public static string CacheRoot =>
        CacheRootOverride ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "hartsyinference");

    /// <summary>The folder a component's libraries for one CUDA major are provisioned into and searched from.</summary>
    public static string CacheLibDir(string component, int cudaMajor) =>
        Path.Combine(CacheRoot, component, $"cuda{cudaMajor}", "lib");

    /// <summary>The optional bundled folder for a component beside the engine assembly (a self-contained deploy may drop libs here).</summary>
    public static string BundledDir(string component) => Path.Combine(AppContext.BaseDirectory, component);

    /// <summary>Fetches <paramref name="spec"/> into its cache lib dir. Returns the lib dir when <see cref="CudaRedistSpec.Sentinel"/>
    /// is present afterwards, else null. Never throws: every failure is logged with the knob that disables the fetch.</summary>
    public static string? FetchInto(CudaRedistSpec spec)
    {
        string libDir = CacheLibDir(spec.Component, spec.CudaMajor);
        // Per run, so two processes provisioning at once (several backends starting on a fresh machine) never share an archive
        // or delete each other's extraction.
        string work = Path.Combine(Path.GetDirectoryName(libDir)!, $"download-{Environment.ProcessId}-{Guid.NewGuid():N}");
        RemoveAbandonedDownloads(Path.GetDirectoryName(libDir)!);
        try
        {
            Directory.CreateDirectory(libDir);
            Directory.CreateDirectory(work);
            Logs.Info($"[{spec.Label}] not found for CUDA {spec.CudaMajor}; fetching {spec.Url} " +
                      $"(one-time, {spec.ApproxSize} → {libDir}; set {spec.KnobHint}=false to disable)...");
            string archive = Path.Combine(work, ArchiveName(spec.Url));
            string sha = Download(spec, archive);
            if (spec.ExpectedSha256 is { } expected && !string.Equals(sha, expected, StringComparison.OrdinalIgnoreCase))
            {
                Logs.Warning($"[{spec.Label}] fetch rejected: sha256 {sha} does not match the expected {expected}. The archive was deleted.");
                return null;
            }
            RunTar($"-xf \"{archive}\" -C \"{work}\"");
            int copied = Flatten(work, libDir, spec.LibraryPatterns);
            bool ok = File.Exists(Path.Combine(libDir, spec.Sentinel));
            if (ok) Logs.Info($"[{spec.Label}] fetch complete: {copied} files → {libDir}.");
            else Logs.Warning($"[{spec.Label}] fetch unpacked {copied} files but {spec.Sentinel} is missing from {libDir}.");
            return ok ? libDir : null;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[{spec.Label}] fetch failed ({ex.Message}). Install it for CUDA {spec.CudaMajor} manually, or set {spec.KnobHint}=false.");
            return null;
        }
        finally
        {
            try { if (Directory.Exists(work)) Directory.Delete(work, recursive: true); }
            catch { /* a leftover download dir is removed by the next fetch once its process has exited */ }
        }
    }

    /// <summary>Deletes download folders a killed run left behind (the per-run <c>download-{pid}-{guid}</c> whose process is gone,
    /// and the legacy fixed <c>download</c>). A folder of a process still running is left alone: it may be mid-fetch.</summary>
    internal static void RemoveAbandonedDownloads(string componentDir)
    {
        if (!Directory.Exists(componentDir)) return;
        foreach (string dir in Directory.EnumerateDirectories(componentDir, "download*"))
        {
            string name = Path.GetFileName(dir);
            string[] parts = name.Split('-');
            bool abandoned = name == "download"
                || (parts.Length == 3 && int.TryParse(parts[1], out int pid) && pid != Environment.ProcessId && !IsRunning(pid));
            if (!abandoned) continue;
            try { Directory.Delete(dir, recursive: true); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* retried on the next fetch */ }
        }
    }

    private static bool IsRunning(int pid)
    {
        try
        {
            using Process process = Process.GetProcessById(pid);
            return !process.HasExited;
        }
        catch (ArgumentException)
        {
            return false;   // no process with that id
        }
        catch (InvalidOperationException)
        {
            return false;
        }
    }

    /// <summary>True when <paramref name="url"/> names a local file rather than an http(s) resource.</summary>
    internal static bool IsLocal(string url) =>
        !Uri.TryCreate(url, UriKind.Absolute, out Uri? uri) || uri.IsFile;

    private static string LocalPath(string url) =>
        Uri.TryCreate(url, UriKind.Absolute, out Uri? uri) && uri.IsFile ? uri.LocalPath : url;

    private static string ArchiveName(string url) =>
        Path.GetFileName(IsLocal(url) ? LocalPath(url) : new Uri(url).LocalPath);

    /// <summary>How long one read may wait for bytes before the download is declared stalled.</summary>
    internal static TimeSpan StallTimeout { get; set; } = TimeSpan.FromMinutes(2);

    /// <summary>Copies the archive to <paramref name="destination"/> in chunks, hashing as it goes, and returns its sha256.
    /// Logs progress every 10 % or 60 s so a long first fetch is visibly alive. Connecting is bounded at 30 s and each read at
    /// <see cref="StallTimeout"/>: <c>HttpClient.Timeout</c> stops at the response headers, so without these a connection that stalls
    /// mid-body would hang the caller (and engine start) forever.</summary>
    private static string Download(CudaRedistSpec spec, string destination)
    {
        using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Stopwatch clock = Stopwatch.StartNew();
        System.Net.Http.HttpClient? http = null;
        System.Net.Http.HttpResponseMessage? response = null;
        try
        {
            Stream source;
            long? total;
            if (IsLocal(spec.Url))
            {
                string path = LocalPath(spec.Url);
                source = File.OpenRead(path);
                total = new FileInfo(path).Length;
            }
            else
            {
                http = new System.Net.Http.HttpClient(new System.Net.Http.SocketsHttpHandler { ConnectTimeout = TimeSpan.FromSeconds(30) })
                {
                    Timeout = TimeSpan.FromMinutes(2),   // until the headers arrive; the body is bounded per read below
                };
                response = http.GetAsync(spec.Url, System.Net.Http.HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
                response.EnsureSuccessStatusCode();
                total = response.Content.Headers.ContentLength ?? spec.ExpectedBytes;
                source = response.Content.ReadAsStream();
            }
            using (source)
            using (FileStream file = File.Create(destination))
            {
                byte[] buffer = new byte[CopyChunk];
                long done = 0;
                int nextDecile = 1;
                TimeSpan lastLog = TimeSpan.Zero;
                int read;
                while ((read = ReadWithStallTimeout(source, buffer)) > 0)
                {
                    file.Write(buffer, 0, read);
                    hash.AppendData(buffer, 0, read);
                    done += read;
                    bool decile = total is > 0 && done * 10 >= total.Value * nextDecile;
                    if (decile || clock.Elapsed - lastLog >= TimeSpan.FromSeconds(60))
                    {
                        if (decile) nextDecile = (int)(done * 10 / total!.Value) + 1;
                        lastLog = clock.Elapsed;
                        string of = total is > 0 ? $"/{total.Value >> 20} MB ({done * 100 / total.Value}%)" : " MB";
                        Logs.Info($"[{spec.Label}] {done >> 20}{of}");
                    }
                }
                Logs.Info($"[{spec.Label}] downloaded {done >> 20} MB in {clock.Elapsed.TotalSeconds:F0} s.");
            }
        }
        finally
        {
            response?.Dispose();
            http?.Dispose();
        }
        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }

    /// <summary>One read that fails with <see cref="TimeoutException"/> when no bytes arrive within <see cref="StallTimeout"/>.</summary>
    internal static int ReadWithStallTimeout(Stream source, byte[] buffer)
    {
        using CancellationTokenSource stall = new(StallTimeout);
        try
        {
            return source.ReadAsync(buffer.AsMemory(), stall.Token).AsTask().GetAwaiter().GetResult();
        }
        catch (OperationCanceledException) when (stall.IsCancellationRequested)
        {
            throw new TimeoutException($"the download stalled: no data for {StallTimeout.TotalSeconds:F0} s");
        }
    }

    /// <summary>Copies every file matching <paramref name="patterns"/> from the extracted tree into <paramref name="libDir"/>.
    /// Real files are copied first; a symlink (Linux sonames) is recreated as a link to its target's file name, so a
    /// <c>libx.so.13 → libx.so.13.10.0.20</c> chain costs one copy of the library, not one per name. NVIDIA's archives keep a
    /// library's links beside it in one folder, which is what the bare-name link assumes.</summary>
    /// <remarks>Each file is written under a temporary name and moved over the final one, so a process that already has the old
    /// library mapped keeps its copy (a rename on Linux) instead of seeing it truncated under it.</remarks>
    internal static int Flatten(string extracted, string libDir, IReadOnlyList<string> patterns)
    {
        List<FileInfo> files = [];
        foreach (string pattern in patterns)
        {
            foreach (string path in Directory.EnumerateFiles(extracted, pattern, SearchOption.AllDirectories))
                files.Add(new FileInfo(path));
        }
        int copied = 0;
        foreach (FileInfo file in files.Where(f => f.LinkTarget is null))
        {
            string destination = Path.Combine(libDir, file.Name);
            string staging = destination + $".{Environment.ProcessId}.tmp";
            File.Copy(file.FullName, staging, overwrite: true);
            try
            {
                File.Move(staging, destination, overwrite: true);
            }
            catch
            {
                // On Windows the move fails while another process has the library loaded; do not leave the staged copy behind.
                try { File.Delete(staging); } catch (IOException) { }
                throw;
            }
            copied++;
        }
        foreach (FileInfo link in files.Where(f => f.LinkTarget is not null))
        {
            string destination = Path.Combine(libDir, link.Name);
            string staging = destination + $".{Environment.ProcessId}.tmp";
            if (File.Exists(staging) || new FileInfo(staging).LinkTarget is not null) File.Delete(staging);
            File.CreateSymbolicLink(staging, Path.GetFileName(link.LinkTarget!));
            File.Move(staging, destination, overwrite: true);
            copied++;
        }
        return copied;
    }

    private static void RunTar(string args)
    {
        using Process process = Process.Start(new ProcessStartInfo("tar", args) { UseShellExecute = false, RedirectStandardError = true })
            ?? throw new InvalidOperationException("failed to start 'tar'");
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0) throw new InvalidOperationException($"tar failed: {stderr}");
    }

    /// <summary>The NVIDIA redist platform name for this OS, or null on an unsupported one.</summary>
    internal static string? Platform() =>
        RuntimeInformation.OSArchitecture != Architecture.X64 ? null
        : RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "windows-x86_64"
        : RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? "linux-x86_64"
        : null;
}
