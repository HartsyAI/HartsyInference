using HartsyInference.Audio.Models.Demucs;
using HartsyInference.Audio.Models.ResembleEnhance;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Audio;

/// <summary>Loads the audio-effects models: Demucs stem separation from a user-placed checkpoint, and
/// Resemble-Enhance speech enhancement from its auto-downloading repo.</summary>
internal static class FxCatalog
{
    /// <summary>Resident Demucs separators, keyed by checkpoint path.</summary>
    internal static readonly AudioRunnerCache<DemucsRunner> DemucsCache = new AudioRunnerCache<DemucsRunner>();

    /// <summary>Resident enhancement pipelines, keyed by repo.</summary>
    internal static readonly AudioRunnerCache<EnhanceRunner> EnhanceCache = new AudioRunnerCache<EnhanceRunner>();

    /// <summary>Sample rate Demucs input is decoded to.</summary>
    internal const int DemucsSampleRate = 44_100;

    /// <summary>Sample rate Resemble-Enhance input is decoded to.</summary>
    internal const int EnhanceSampleRate = 44_100;

    /// <summary>The Resemble-Enhance weights repo.</summary>
    internal const string EnhanceRepo = "ResembleAI/resemble-enhance";

    /// <summary>Resolves the Demucs checkpoint path for a variant name (default <c>htdemucs</c>), accepting a direct
    /// file name, a <c>.th</c>, or a <c>.safetensors</c> in the fx weights folder.</summary>
    internal static string ResolveDemucsPath(string? modelName, string? localPath)
    {
        if (!string.IsNullOrEmpty(localPath) && File.Exists(localPath))
        {
            return localPath;
        }
        string directory = AudioModelRoot.WeightsDirectory("fx", "demucs");
        string variant = string.IsNullOrWhiteSpace(modelName) ? "htdemucs" : modelName.Trim();
        string direct = Path.Combine(directory, variant);
        if (File.Exists(direct))
        {
            return direct;
        }
        string torch = Path.Combine(directory, variant + ".th");
        if (File.Exists(torch))
        {
            return torch;
        }
        string safeTensors = Path.Combine(directory, variant + ".safetensors");
        return File.Exists(safeTensors) ? safeTensors : torch;   // 'th' is used in the not-found message
    }

    /// <summary>Loads a Demucs checkpoint, picking the architecture config from the model name.</summary>
    internal static DemucsRunner LoadDemucs(string path, string? modelName)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"Demucs weights not found: '{path}'. Place the htdemucs checkpoint (.th or .safetensors) there.", path);
        }
        (IReadOnlyDictionary<string, Tensor> weights, IDisposable loader) = AudioCheckpoints.LoadFile(path);
        HtDemucsConfig config = ConfigFor(modelName);
        DemucsPipeline pipeline = new DemucsPipeline(config);
        pipeline.LoadWeights(weights);
        Logs.Info($"[Audio][Demucs] Loaded '{Path.GetFileName(path)}' ({config.NumSources} stems, 44.1 kHz).");
        return new DemucsRunner(pipeline, loader);
    }

    /// <summary>Loads the Resemble-Enhance denoiser + LCFM enhancer + UnivNet vocoder.</summary>
    internal static async Task<EnhanceRunner> LoadEnhanceAsync(CancellationToken cancel)
    {
        (IReadOnlyDictionary<string, Tensor> weights, IDisposable[] loaders) = await AudioCheckpoints.LoadAsync(EnhanceRepo, cancel).ConfigureAwait(false);
        ResembleEnhancePipeline pipeline = new ResembleEnhancePipeline(ResembleEnhanceConfig.Default, withDenoiserAndVocoder: true);
        pipeline.LoadWeights(weights);
        Logs.Info("[Audio][Resemble-Enhance] Loaded ResembleAI/resemble-enhance (denoiser + LCFM + UnivNet, 44.1 kHz).");
        return new EnhanceRunner(pipeline, loaders);
    }

    /// <summary>Selects the architecture config from the model name: the 6-stem variant adds guitar+piano and MUST
    /// use the 6-source config or the final decoder shape mismatches the weights on load.</summary>
    private static HtDemucsConfig ConfigFor(string? modelName)
    {
        string value = (modelName ?? string.Empty).Trim().ToLowerInvariant();
        if (value.Contains("6s", StringComparison.Ordinal))
        {
            return HtDemucsConfig.Htdemucs6s;
        }
        return value.Contains("_ft", StringComparison.Ordinal) || value.EndsWith("ft", StringComparison.Ordinal)
            ? HtDemucsConfig.HtdemucsFt
            : HtDemucsConfig.Htdemucs;
    }
}
