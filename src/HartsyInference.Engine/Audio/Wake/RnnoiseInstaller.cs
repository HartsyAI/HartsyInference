using System.Security.Cryptography;
using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio.Wake;

/// <summary>Installs the RNNoise weights <see cref="WakeModelSet.LoadDenoiser"/> loads, from xiph's own release.
///
/// <para>Fetches the model tarball upstream's <c>download_model.sh</c> fetches, rejects it unless its SHA-256 is the
/// one upstream pins (its <c>model_version</c> file, which also names the tarball), and converts the checkpoint in C#
/// through <see cref="RnnoiseCheckpoint"/>. Nothing is re-hosted, so media.xiph.org is the only source.</para>
///
/// <para>This is an explicit install and always reaches the network. A caller fetching on its own initiative should
/// check <c>paths.sideModelAutofetch</c> first.</para></summary>
public static class RnnoiseInstaller
{
    /// <summary>SHA-256 of the model tarball: xiph/rnnoise's <c>model_version</c> as of commit 70f1d25.</summary>
    public const string TarballSha256 = "0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37";

    /// <summary>Where upstream's <c>download_model.sh</c> fetches the tarball from.</summary>
    public const string TarballUrl = "https://media.xiph.org/rnnoise/models/rnnoise_data-" + TarballSha256 + ".tar.gz";

    private static readonly SemaphoreSlim _gate = new(1, 1);

    /// <summary>The denoiser's path under a wake model root.</summary>
    public static string WeightsPath(string modelRoot) => Path.Combine(modelRoot, "denoise", "rnnoise.safetensors");

    /// <summary>Downloads, verifies and converts the denoiser into <paramref name="modelRoot"/> unless it is already
    /// installed, and returns its path. The tarball is deleted once converted. A file already at the path is taken
    /// as installed without being read; delete it to force a reinstall.</summary>
    public static async Task<string> EnsureAsync(string modelRoot, CancellationToken cancel)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelRoot);
        string target = WeightsPath(modelRoot);
        await _gate.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            if (File.Exists(target))
            {
                return target;
            }
            string tarball = Path.Combine(Path.GetDirectoryName(target)!, $"rnnoise_data-{TarballSha256}.tar.gz");
            try
            {
                Logs.Info($"[Audio][Wake] Downloading the RNNoise model from '{TarballUrl}' (one-time, ~59 MB)...");
                await AudioFileFetcher.EnsureAsync(TarballUrl, tarball, TarballSha256, cancel).ConfigureAwait(false);
                RnnoiseCheckpoint.ConvertTarball(tarball, target, Metadata());
            }
            finally
            {
                File.Delete(tarball);
            }
            Logs.Info($"[Audio][Wake] RNNoise denoiser installed at '{target}'.");
            return target;
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>Installs from a copy of the tarball obtained elsewhere, for a machine without network access, and
    /// returns the weights path. The copy must be the pinned release; an existing install is replaced.</summary>
    /// <exception cref="InvalidDataException">The tarball's SHA-256 is not <see cref="TarballSha256"/>.</exception>
    public static string InstallFromTarball(string tarballPath, string modelRoot)
    {
        ArgumentException.ThrowIfNullOrEmpty(tarballPath);
        ArgumentException.ThrowIfNullOrEmpty(modelRoot);
        string actual;
        using (FileStream stream = File.OpenRead(tarballPath))
        {
            actual = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        }
        if (!string.Equals(actual, TarballSha256, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"'{tarballPath}' has SHA-256 {actual}, not the pinned RNNoise release {TarballSha256}.");
        }
        string target = WeightsPath(modelRoot);
        _gate.Wait();
        try
        {
            RnnoiseCheckpoint.ConvertTarball(tarballPath, target, Metadata());
        }
        finally
        {
            _gate.Release();
        }
        return target;
    }

    /// <summary>Provenance for the header. <c>tools/convert_rnnoise.py</c> writes the same keys, so a file from
    /// either converter says where it came from in the same words.</summary>
    private static Dictionary<string, string> Metadata() => new(StringComparer.Ordinal)
    {
        ["hartsy.component"] = "denoiser",
        ["hartsy.converter"] = "HartsyInference.RnnoiseCheckpoint",
        ["hartsy.license"] = RnnoiseCheckpoint.License,
        ["hartsy.source_url"] = TarballUrl,
        ["hartsy.source_sha256"] = TarballSha256,
        ["hartsy.source_member"] = RnnoiseCheckpoint.CheckpointMember,
    };
}
