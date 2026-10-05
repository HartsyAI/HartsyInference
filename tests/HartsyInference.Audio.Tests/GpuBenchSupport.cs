using System.Globalization;
using System.Text;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Configuration;
using HartsyInference.Cuda;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>What the opt-in 3060 audio benches share: opening the device they assert on, the model files they gate on,
/// emitting their tables, and the percentile they report.</summary>
internal static class GpuBenchSupport
{
    /// <summary>The device every audio bench in this project asserts it runs on.</summary>
    public const string RequiredDeviceSubstring = "3060";

    public const string KokoroRepackRepo = "Hartsy/kokoro-82m-safetensors";
    public const string KokoroRepo = "hexgrad/Kokoro-82M";
    public const string KokoroVoice = "af_heart";

    /// <summary>Opens the CUDA device named by <paramref name="ordinalEnvVar"/> (default 1, the 3060's engine ordinal on
    /// the reference box: CUDA enumerates fastest-first, the opposite of nvidia-smi) and fails unless it is a 3060.</summary>
    public static CudaBackend Open3060(Action<string> log, string ordinalEnvVar)
    {
        string? ordinalText = Environment.GetEnvironmentVariable(ordinalEnvVar);
        int ordinal = string.IsNullOrEmpty(ordinalText) ? 1 : int.Parse(ordinalText, CultureInfo.InvariantCulture);
        Assert.True(CudaContext.IsAvailable(), $"CUDA unavailable: {CudaContext.LastUnavailableReason}");
        string? ptx = BackendGate.KernelDir("Ptx", "HartsyInference.Cuda");
        Assert.False(ptx is null, "no compiled PTX directory beside the tests or in the repo");
        CudaBackend backend = new CudaBackend(ordinal, ptx);
        string device = backend.Capabilities.DeviceName;
        log($"CUDA ordinal {ordinal}: {device}; models root {ModelsRoot()}; audio cache {AudioModelCache.CacheRoot}");
        if (!device.Contains(RequiredDeviceSubstring, StringComparison.Ordinal))
        {
            backend.Dispose();
            Assert.Fail($"ordinal {ordinal} is '{device}', not a {RequiredDeviceSubstring}. Set {ordinalEnvVar} to the "
                + "3060's engine ordinal.");
        }
        return backend;
    }

    public static string ModelsRoot() =>
        EngineKnobs.ModelsRoot.Value is { Length: > 0 } root ? Path.GetFullPath(root) : TestPaths.ModelsDir;

    /// <summary>The files the engine's Kokoro G2P reads: misaki's two dictionaries, then the CMU dictionary.</summary>
    public static string[] KokoroG2PFiles()
    {
        string audio = Path.Combine(ModelsRoot(), "audio");
        return [Path.Combine(audio, "misaki_us_gold.json"), Path.Combine(audio, "misaki_us_silver.json"),
            Path.Combine(audio, "cmudict.dict")];
    }

    /// <summary>The engine's Kokoro G2P over <see cref="KokoroG2PFiles"/> (no espeak fallback).</summary>
    public static EnglishG2P KokoroG2P()
    {
        string[] files = KokoroG2PFiles();
        return new EnglishG2P(MisakiLexicon.FromFiles(files[0], files[1]), files[2]);
    }

    public static string[] WhisperFiles(string repo)
    {
        string dir = AudioModelCache.GetRepoDirectory(repo, "stt");
        return WhisperPipeline.ModelFiles.Where(f => f.Required).Select(f => Path.Combine(dir, f.Name)).ToArray();
    }

    public static string[] KokoroFiles()
    {
        string repack = AudioModelCache.GetRepoDirectory(KokoroRepackRepo, "tts");
        string canonical = AudioModelCache.GetRepoDirectory(KokoroRepo, "tts");
        return
        [
            Path.Combine(repack, "kokoro-82m.safetensors"),
            Path.Combine(canonical, "config.json"),
            Path.Combine(canonical, "voices", KokoroVoice + ".bin"),
        ];
    }

    /// <summary>Writes <paramref name="table"/> to the test log and, when <paramref name="outEnvVar"/> names a file,
    /// appends it there.</summary>
    public static void Emit(Action<string> log, StringBuilder table, string outEnvVar)
    {
        string text = table.ToString();
        log(text);
        string? outPath = Environment.GetEnvironmentVariable(outEnvVar);
        if (!string.IsNullOrEmpty(outPath))
        {
            File.AppendAllText(outPath, text + Environment.NewLine);
            log($"appended to {outPath}");
        }
    }

    /// <summary>Linear-interpolated percentile of an ascending list (p in [0, 1]).</summary>
    public static double Percentile(IReadOnlyList<double> sorted, double p)
    {
        if (sorted.Count == 1)
        {
            return sorted[0];
        }
        double position = p * (sorted.Count - 1);
        int low = (int)Math.Floor(position);
        int high = Math.Min(low + 1, sorted.Count - 1);
        return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
    }
}
