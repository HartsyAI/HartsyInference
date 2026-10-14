using System.Collections.Concurrent;
using System.IO.Compression;
using System.Security.Cryptography;
using HartsyInference.Audio.Frontends.WeText;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>The wetext text-normalization grammars (pengzhendong/wetext, Apache-2.0), extracted once from the
/// <c>wetext</c> 0.1.8 wheel on PyPI (checked against its SHA-256) into <c>{audio}/wetext/{lang}/tn/*.fst</c>, each
/// grammar checked against its own SHA-256. wetext 0.0.4, which CosyVoice pins, downloads the same files from
/// ModelScope <c>pengzhendong/wetext</c> master at runtime; that repository rate-limits bursts, the PyPI CDN does not.
/// These are the files <c>wetext_parity.json</c> was generated from.</summary>
internal static class WeTextAssets
{
    private const string WheelUrl = "https://files.pythonhosted.org/packages/43/fe/"
        + "ca7ccae2673b64ba7d63612e68b31963e3842dd77fb6aa632270d123d685/wetext-0.1.8-py3-none-any.whl";
    private const string WheelSha256 = "b2083e7f38ac38fcbecdf7aec6ac6f0e3d1a11b763facfa4daabc12138215454";
    private const string WheelEntryPrefix = "wetext/fsts/";

    /// <summary>The folder under the audio root the grammars land in.</summary>
    internal const string Folder = "wetext";

    private static readonly ConcurrentDictionary<string, SemaphoreSlim> Gates = new(StringComparer.Ordinal);

    private static readonly Dictionary<string, string> Sha256 =new(StringComparer.Ordinal)
    {
        ["en/tn/tagger.fst"] = "245e2dc9174cdd007a8e9e50f3339773d1adbbc7535b71cf67478dc1683cc3ec",
        ["en/tn/verbalizer.fst"] = "03155c88f317b2795969e264c19f87faf98b9853d5ac631e419bacdc3b3ee15a",
        ["zh/tn/tagger.fst"] = "cf341314c51f7ce59049f3b2c42f0ce8fd71e6d08d4d6969613aad384a5e2ae8",
        ["zh/tn/verbalizer.fst"] = "5a13cd679dd54637d12d2bd1bd33ee2165d91c867e14468c93195af02256e5da",
        ["zh/tn/verbalizer_remove_erhua.fst"] = "4ecdea636b97cb7dff2637499713815836ec703d29cf044f17106d428f258ee5",
    };

    /// <summary>Extracts any missing grammar for <paramref name="language"/> and returns the folder holding them.</summary>
    public static async Task<string> EnsureAsync(WeTextLanguage language, bool removeErhua, CancellationToken cancel)
    {
        string directory = AudioModelRoot.SharedFile(Folder);
        if (Missing(directory, language, removeErhua).Count == 0) return directory;

        SemaphoreSlim gate = Gates.GetOrAdd(Path.GetFullPath(directory), _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            // Another caller may have extracted them while this one waited.
            List<string> missing = Missing(directory, language, removeErhua);
            if (missing.Count == 0) return directory;
            // A per-call wheel name keeps another process's fetch of the same folder from colliding with this one.
            string wheel = Path.Combine(directory, $"{Path.GetFileName(WheelUrl)}.{Guid.NewGuid():N}");
            try
            {
                Logs.Info("[Audio][WeText] Downloading the text-normalization grammars (wetext 0.1.8 wheel, 5 MB)...");
                await AudioFileFetcher.EnsureAsync(WheelUrl, wheel, WheelSha256, cancel).ConfigureAwait(false);
                await Task.Run(() => Extract(wheel, directory, missing), cancel).ConfigureAwait(false);
            }
            finally
            {
                File.Delete(wheel);
            }
            return directory;
        }
        finally
        {
            gate.Release();
        }
    }

    /// <summary>Extracts any missing grammar and loads the normalizer from them.</summary>
    public static async Task<WeTextNormalizer> LoadAsync(WeTextLanguage language, bool removeErhua,
        CancellationToken cancel)
    {
        string directory = await EnsureAsync(language, removeErhua, cancel).ConfigureAwait(false);
        return await Task.Run(() => WeTextNormalizer.Load(directory, language, removeErhua), cancel).ConfigureAwait(false);
    }

    private static List<string> Missing(string directory, WeTextLanguage language, bool removeErhua) =>
        [.. WeTextNormalizer.RequiredFiles(language, removeErhua).Where(f => !File.Exists(Path.Combine(directory, f)))];

    // Staged through a uniquely named temp file and moved into place only once its SHA-256 matches.
    private static void Extract(string wheel, string directory, List<string> files)
    {
        using ZipArchive zip = ZipFile.OpenRead(wheel);
        foreach (string file in files)
        {
            ZipArchiveEntry entry = zip.GetEntry(WheelEntryPrefix + file)
                ?? throw new InvalidDataException($"{Path.GetFileName(wheel)} has no {WheelEntryPrefix}{file}");
            string target = Path.Combine(directory, file);
            string temp = $"{target}.{Guid.NewGuid():N}.tmp";
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            try
            {
                entry.ExtractToFile(temp, overwrite: false);
                byte[] hash;
                using (FileStream stream = File.OpenRead(temp)) hash = SHA256.HashData(stream);
                string actual = Convert.ToHexString(hash).ToLowerInvariant();
                if (actual != Sha256[file])
                    throw new InvalidDataException($"wetext grammar {file} has SHA-256 {actual}, expected {Sha256[file]}");
                File.Move(temp, target, overwrite: true);
            }
            finally
            {
                File.Delete(temp);
            }
        }
    }
}
