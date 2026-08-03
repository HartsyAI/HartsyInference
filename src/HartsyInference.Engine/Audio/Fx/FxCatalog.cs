using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Demucs;
using HartsyInference.Audio.Models.ResembleEnhance;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.Engine.Audio;

/// <summary>Loads the audio-effects models: Demucs stem separation and Resemble-Enhance speech enhancement,
/// both from auto-downloading sources (Demucs isn't on HuggingFace as a single-file checkpoint — its official
/// weights are fetched directly from Meta's public CDN).</summary>
internal static class FxCatalog
{
    /// <summary>Sample rate Demucs input is decoded to.</summary>
    internal const int DemucsSampleRate = 44_100;

    /// <summary>Sample rate Resemble-Enhance input is decoded to.</summary>
    internal const int EnhanceSampleRate = 44_100;

    /// <summary>The Resemble-Enhance weights repo.</summary>
    internal const string EnhanceRepo = "ResembleAI/resemble-enhance";

    /// <summary>Official Meta CDN URLs for the two single-checkpoint HTDemucs variants (confirmed public, no HF
    /// gating, no auth — resolved from the upstream <c>demucs/remote/*.yaml</c>/<c>files.txt</c> manifests: model
    /// signature → hash-named <c>.th</c> under <c>https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/</c>).
    /// <c>htdemucs_ft</c> is deliberately excluded — upstream ships it as a 4-checkpoint <c>Bag_of_models</c>
    /// ensemble (one single-source model per stem, weight-averaged), not a single 4-stem checkpoint like the other
    /// two; wiring it needs ensemble support this pipeline doesn't have. It stays manual-placement (<c>--model-path</c>).</summary>
    private static readonly Dictionary<string, string> DemucsCdnUrls = new(StringComparer.OrdinalIgnoreCase)
    {
        ["htdemucs"] = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
        ["htdemucs_6s"] = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
    };

    /// <summary>Resolves the Demucs checkpoint path for a variant name (default <c>htdemucs</c>), auto-downloading
    /// the official Meta checkpoint on first use unless a local file (direct <paramref name="localPath"/>, or a
    /// user-placed <c>.th</c>/<c>.safetensors</c> already in the fx weights folder) is present.</summary>
    internal static async Task<string> EnsureDemucsPathAsync(string? modelName, string? localPath, CancellationToken cancel)
    {
        if (!string.IsNullOrEmpty(localPath) && File.Exists(localPath))
        {
            return localPath;
        }
        string directory = AudioModelRoot.WeightsDirectory("fx", "demucs");
        // Bare "-m demucs" (no ":variant") resolves modelName to the literal catalog id "demucs", not a real
        // htdemucs variant name (AudioModelSelector.Parse passes the whole token through when there's no colon) —
        // treat that the same as "unset" so it maps to the actual default variant instead of looking for a
        // nonexistent "demucs.th".
        string trimmed = (modelName ?? string.Empty).Trim();
        string variant = string.IsNullOrEmpty(trimmed) || trimmed.Equals("demucs", StringComparison.OrdinalIgnoreCase)
            ? "htdemucs" : trimmed;
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
        if (File.Exists(safeTensors))
        {
            return safeTensors;
        }
        if (DemucsCdnUrls.TryGetValue(variant, out string? url))
        {
            Logs.Info($"[Audio][Demucs] Downloading the official '{variant}' checkpoint from Meta's CDN (one-time)...");
            await AudioFileFetcher.EnsureAsync(url, torch, cancel).ConfigureAwait(false);
            Logs.Info($"[Audio][Demucs] '{variant}' checkpoint ready.");
            return torch;
        }
        return torch;   // 'th' is used in the not-found message for variants with no known auto-download source (htdemucs_ft).
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

    /// <summary>Path of the DeepSpeed generator checkpoint inside the HF repo — the repo ships no
    /// model.safetensors / pytorch_model.bin, only the raw <c>enhancer_stage2</c> run directory.</summary>
    private const string EnhanceCheckpointFile = "enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt";

    /// <summary>Loads the Resemble-Enhance denoiser + LCFM enhancer + UnivNet vocoder from the repo's DeepSpeed
    /// <c>mp_rank_00_model_states.pt</c> (~700 MB, downloaded on first use), with a strict zero-missing/
    /// zero-unexpected key check inside <c>LoadWeights</c>.</summary>
    internal static async Task<EnhanceRunner> LoadEnhanceAsync(CancellationToken cancel)
    {
        string path = await AudioModelCache.GetAsync(EnhanceRepo, EnhanceCheckpointFile, category: "fx", ct: cancel).ConfigureAwait(false);
        (IReadOnlyDictionary<string, Tensor> weights, IDisposable loader) = DeepSpeedCheckpointConverter.Load(path);
        ResembleEnhancePipeline pipeline = new ResembleEnhancePipeline(ResembleEnhanceConfig.Default, withDenoiserAndVocoder: true);
        pipeline.LoadWeights(weights);
        Logs.Info($"[Audio][Resemble-Enhance] Loaded {EnhanceRepo} (denoiser + LCFM + UnivNet, {weights.Count} tensors, 44.1 kHz).");
        return new EnhanceRunner(pipeline, [loader]);
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
