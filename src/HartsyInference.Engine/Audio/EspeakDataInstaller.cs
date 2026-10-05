using System.IO.Compression;
using HartsyInference.Audio.Phonemizer.Espeak;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Installs <c>espeak-ng-data</c> into the model cache, where <see cref="EspeakPhonemizer.FromCache"/> finds
/// it: the copy inside the pinned <c>espeakng_loader</c> 0.2.4 wheel (espeak-ng 1.52, GPL-3.0), the data misaki and
/// the reference Kokoro pipeline phonemize with. Only the data directory is unpacked; the wheel is deleted after.
/// A target directory left half-written by an older install (no <c>phontab</c>) makes the publish fail, so that
/// case is cleared first.</summary>
internal static class EspeakDataInstaller
{
    private const string WheelUrl = "https://files.pythonhosted.org/packages/9d/ed/a3d872fbad4f3a3f3db0e8c31768ab14e77cd77306de16b8b20b1e1df7ea/espeakng_loader-0.2.4-py3-none-win_amd64.whl";
    private const string WheelSha256 = "41f1e08ac9deda2efd1ea9de0b81dab9f5ae3c4b24284f76533d0a7b1dd7abd7";
    private const string DataPrefix = "espeakng_loader/espeak-ng-data/";

    private static readonly SemaphoreSlim _gate = new(1, 1);

    /// <summary>Installs the data unless the model cache already holds it, and returns the directory to read. An
    /// <c>ESPEAK_DATA_DIR</c> override is used as given. Without network access, any espeak-ng data
    /// <see cref="EspeakPhonemizer.FromCache"/> can find (a system install) is used instead, with a warning, since
    /// an older release reads some words differently from what Kokoro was trained on.</summary>
    public static async Task<string> EnsureAsync(CancellationToken cancel)
    {
        string? overridden = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (!string.IsNullOrEmpty(overridden) && File.Exists(Path.Combine(overridden, "phontab"))) return overridden;
        string target = EspeakPhonemizer.CacheDataDirectory;
        if (File.Exists(Path.Combine(target, "phontab"))) return target;
        await _gate.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            if (File.Exists(Path.Combine(target, "phontab"))) return target;
            try
            {
                await InstallAsync(target, cancel).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is HttpRequestException or IOException or InvalidDataException or UnauthorizedAccessException)
            {
                if (!EspeakPhonemizer.TryFindDataDirectory(out string? existing)) throw;
                Logs.Warning($"[Audio] Could not install espeak-ng 1.52 data ({ex.Message}); using '{existing}'.");
                return existing;
            }
            return target;
        }
        finally
        {
            _gate.Release();
        }
    }

    // Downloads and unpacks into a staging directory unique to this call, then publishes it with one directory move,
    // so engines in other processes sharing the cache never see, or delete, a half-written copy.
    private static async Task InstallAsync(string target, CancellationToken cancel)
    {
        string parent = Path.GetDirectoryName(target)!;
        Directory.CreateDirectory(parent);
        string unique = Guid.NewGuid().ToString("N");
        string wheel = Path.Combine(parent, $"espeakng_loader-{unique}.whl");
        string staging = Path.Combine(parent, $"espeak-ng-data.{unique}.tmp");
        try
        {
            Logs.Info("[Audio] Downloading espeak-ng data (one-time, ~9 MB)...");
            await AudioFileFetcher.EnsureAsync(WheelUrl, wheel, WheelSha256, cancel).ConfigureAwait(false);
            Extract(wheel, staging);
            if (Directory.Exists(target) && !File.Exists(Path.Combine(target, "phontab")))
                Directory.Delete(target, recursive: true); // not a copy this installer published
            try
            {
                Directory.Move(staging, target);
            }
            catch (IOException) when (File.Exists(Path.Combine(target, "phontab")))
            {
                return; // another engine published it first
            }
            Logs.Info($"[Audio] espeak-ng data installed at '{target}'.");
        }
        finally
        {
            if (File.Exists(wheel)) File.Delete(wheel);
            if (Directory.Exists(staging)) Directory.Delete(staging, recursive: true);
        }
    }

    // Unpacks the wheel's espeak-ng-data tree into directory, refusing entries that would land outside it.
    private static void Extract(string wheel, string directory)
    {
        string root = Path.GetFullPath(directory) + Path.DirectorySeparatorChar;
        using ZipArchive zip = ZipFile.OpenRead(wheel);
        foreach (ZipArchiveEntry entry in zip.Entries)
        {
            if (!entry.FullName.StartsWith(DataPrefix, StringComparison.Ordinal) || entry.FullName.EndsWith('/')) continue;
            string path = Path.GetFullPath(Path.Combine(directory, entry.FullName[DataPrefix.Length..]));
            if (!path.StartsWith(root, StringComparison.Ordinal))
                throw new InvalidDataException($"The espeak-ng wheel entry '{entry.FullName}' escapes its directory.");
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            entry.ExtractToFile(path, overwrite: true);
        }
        if (!File.Exists(Path.Combine(directory, "phontab")))
            throw new InvalidDataException("The espeak-ng wheel holds no espeak-ng-data/phontab.");
    }
}
