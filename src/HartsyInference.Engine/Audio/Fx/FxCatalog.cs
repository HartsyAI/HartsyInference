using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Demucs;
using HartsyInference.Audio.Models.ResembleEnhance;
using HartsyInference.Audio.Models.Separation;
using HartsyInference.Audio.Models.VoiceIsolator;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.Engine.Audio;

/// <summary>Loads the audio-effects models: Demucs stem separation and Resemble-Enhance speech enhancement, both from auto-downloading sources (Demucs isn't on HuggingFace as a single-file checkpoint — its official weights are fetched directly from Meta's public CDN).</summary>
internal static class FxCatalog
{
    /// <summary>Sample rate Demucs input is decoded to.</summary>
    internal const int DemucsSampleRate = 44_100;

    /// <summary>Sample rate Resemble-Enhance input is decoded to.</summary>
    internal const int EnhanceSampleRate = 44_100;

    /// <summary>Catalog id of the Kim_Vocal_2 vocal separator (<c>-m kim-vocal-2</c>).</summary>
    internal const string KimVocal2Id = "kim-vocal-2";

    /// <summary>Where Kim_Vocal_2 ships: the vocal_separator folder of the LongCat-Video-Avatar-1.6 repo, 66,759,214 bytes.</summary>
    internal const string KimVocal2Repo = "meituan-longcat/LongCat-Video-Avatar-1.6";

    /// <summary>The file's path inside <see cref="KimVocal2Repo"/>.</summary>
    internal const string KimVocal2File = "vocal_separator/Kim_Vocal_2.onnx";

    /// <summary>sha256 of <see cref="KimVocal2File"/>, the repo's LFS oid.</summary>
    internal const string KimVocal2Sha256 = "ce74ef3b6a6024ce44211a07be9cf8bc6d87728cc852a68ab34eb8e58cde9c8b";

    /// <summary>Catalog id of the offline Inworld Voice Isolator (<c>-m voice-isolator</c>).</summary>
    internal const string VoiceIsolatorId = "voice-isolator";

    /// <summary>Catalog id of the causal, streaming Inworld Voice Isolator (<c>-m voice-isolator-realtime</c>).</summary>
    internal const string VoiceIsolatorRealtimeId = "voice-isolator-realtime";

    /// <summary>Look-ahead positions the realtime checkpoint was trained at, in frames of 10 ms.</summary>
    internal static readonly int[] VoiceIsolatorRealtimeLookaheads = [0, 2, 4];

    /// <summary>Sample rate the voice isolator runs at.</summary>
    internal const int VoiceIsolatorSampleRate = 48_000;

    /// <summary>The weights repo of both voice isolators (Apache-2.0).</summary>
    internal const string VoiceIsolatorRepo = "inworld/voice-isolator";

    /// <summary>Repo revision the file sizes and hashes below were read from.</summary>
    internal const string VoiceIsolatorRevision = "9b0285b3fe8403aa8e2f1dbb2165a80a4294c970";

    /// <summary>Offline checkpoint: 256,260,128 bytes, 64.0M parameters.</summary>
    internal const string VoiceIsolatorFile = "denoiser.safetensors";

    /// <summary>sha256 of <see cref="VoiceIsolatorFile"/>.</summary>
    internal const string VoiceIsolatorSha256 = "647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53";

    /// <summary>Realtime checkpoint: 232,780,800 bytes, 58.2M parameters.</summary>
    internal const string VoiceIsolatorRealtimeFile = "denoiser_realtime.safetensors";

    /// <summary>sha256 of <see cref="VoiceIsolatorRealtimeFile"/>.</summary>
    internal const string VoiceIsolatorRealtimeSha256 = "76598f05523c0669e76562a8b2f31c9d395fca316e80c907ee11775e7497ae0f";

    /// <summary>The Resemble-Enhance weights repo.</summary>
    internal const string EnhanceRepo = "ResembleAI/resemble-enhance";

    /// <summary>Official Meta CDN URLs for the two single-checkpoint HTDemucs variants (confirmed public, no HF gating, no auth — resolved from the upstream <c>demucs/remote/*.yaml</c>/<c>files.txt</c> manifests: model signature → hash-named <c>.th</c> under <c>https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/</c>). <c>htdemucs_ft</c> is deliberately excluded — upstream ships it as a 4-checkpoint <c>Bag_of_models</c> ensemble (one single-source model per stem, weight-averaged), not a single 4-stem checkpoint like the other two; wiring it needs ensemble support this pipeline doesn't have. It stays manual-placement (<c>--model-path</c>).</summary>
    private static readonly Dictionary<string, string> DemucsCdnUrls = new(StringComparer.OrdinalIgnoreCase)
    {
        ["htdemucs"] = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
        ["htdemucs_6s"] = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
    };

    /// <summary>Resolves the Demucs checkpoint path for a variant name (default <c>htdemucs</c>), auto-downloading the official Meta checkpoint on first use unless a local file (direct <paramref name="localPath"/>, or a user-placed <c>.th</c>/<c>.safetensors</c> already in the fx weights folder) is present.</summary>
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

    /// <summary>Resolves Kim_Vocal_2's <c>.onnx</c>: an explicit local path, else a file already in the fx weights folder, else the repo copy
    /// (downloaded once into the audio cache and checked against <see cref="KimVocal2Sha256"/>).</summary>
    internal static async Task<string> EnsureKimVocal2PathAsync(string? localPath, CancellationToken cancel)
    {
        if (!string.IsNullOrEmpty(localPath) && File.Exists(localPath))
        {
            return localPath;
        }
        string placed = Path.Combine(AudioModelRoot.WeightsDirectory("fx", KimVocal2Id), Path.GetFileName(KimVocal2File));
        if (File.Exists(placed))
        {
            return placed;
        }
        string? cached = AudioModelCache.TryGetLocal(KimVocal2Repo, KimVocal2File, "fx");
        if (cached is not null)
        {
            return cached;
        }
        Logs.Info($"[Audio][Kim-Vocal-2] Downloading {KimVocal2File} from {KimVocal2Repo} (66.8 MB, one-time)...");
        string path = await AudioModelCache.GetAsync(KimVocal2Repo, KimVocal2File, "fx", ct: cancel).ConfigureAwait(false);
        AudioModelCache.VerifySha256(path, KimVocal2Sha256);
        return path;
    }

    /// <summary>Reads Kim_Vocal_2's graph and weights from <paramref name="path"/> (a repack to safetensors is a later step: the file is 66.8 MB and binds in
    /// about a tenth of a second) and builds the runner on <paramref name="backend"/>.</summary>
    internal static MdxVocalRunner LoadKimVocal2(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Kim_Vocal_2 weights not found: '{path}'. Place Kim_Vocal_2.onnx there.", path);
        }
        Dictionary<string, Tensor> weights = KimVocal2OnnxConverter.Convert(path, out KimVocal2Geometry geometry);
        try
        {
            KimVocal2Network network = new(weights);
            Logs.Info($"[Audio][Kim-Vocal-2] Loaded '{Path.GetFileName(path)}' ({geometry.Levels} levels, widths {string.Join('/', geometry.Widths)}, "
                + $"{geometry.DimF} bins, F32).");
            return new MdxVocalRunner(network, weights);
        }
        catch
        {
            foreach (Tensor t in weights.Values)
            {
                t.Dispose();
            }
            throw;
        }
    }

    /// <summary>Resolves a voice isolator checkpoint: an explicit local path, else the file already in the fx weights folder, else the
    /// repo copy (downloaded once and checked against its pinned sha256).</summary>
    internal static async Task<string> EnsureVoiceIsolatorPathAsync(bool realtime, string? localPath, CancellationToken cancel)
    {
        if (!string.IsNullOrEmpty(localPath) && File.Exists(localPath))
        {
            return localPath;
        }
        string file = realtime ? VoiceIsolatorRealtimeFile : VoiceIsolatorFile;
        string sha = realtime ? VoiceIsolatorRealtimeSha256 : VoiceIsolatorSha256;
        string placed = Path.Combine(AudioModelRoot.WeightsDirectory("fx", VoiceIsolatorId), file);
        if (File.Exists(placed))
        {
            return placed;
        }
        string? cached = AudioModelCache.TryGetLocal(VoiceIsolatorRepo, file, "fx");
        if (cached is not null)
        {
            return cached;
        }
        long megabytes = realtime ? 233 : 256;
        Logs.Info($"[Audio][Voice-Isolator] Downloading {file} from {VoiceIsolatorRepo} ({megabytes} MB, one-time)...");
        string path = await AudioModelCache.GetAsync(VoiceIsolatorRepo, file, "fx", VoiceIsolatorRevision, ct: cancel).ConfigureAwait(false);
        AudioModelCache.VerifySha256(path, sha);
        return path;
    }

    /// <summary>Reads a voice isolator checkpoint; its architecture comes from the file's own metadata.</summary>
    internal static VoiceIsolatorRunner LoadVoiceIsolator(string path, bool realtime)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Voice isolator weights not found: '{path}'.", path);
        }
        VoiceIsolatorNetwork network = VoiceIsolatorNetwork.Load(path);
        if (network.Config.Causal != realtime)
        {
            network.Dispose();
            throw new ArgumentException($"'{Path.GetFileName(path)}' is the {(network.Config.Causal ? "realtime" : "offline")} checkpoint; the requested model is the {(realtime ? "realtime" : "offline")} one.", nameof(path));
        }
        Logs.Info($"[Audio][Voice-Isolator] Loaded '{Path.GetFileName(path)}' ({(network.Config.Causal ? "realtime" : "offline")}, "
            + $"{network.BandCount} bands, width {network.Config.Width}, F32).");
        return new VoiceIsolatorRunner(network);
    }

    /// <summary>Path of the DeepSpeed generator checkpoint inside the HF repo — the repo ships no model.safetensors / pytorch_model.bin, only the raw <c>enhancer_stage2</c> run directory.</summary>
    private const string EnhanceCheckpointFile = "enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt";

    /// <summary>Loads the Resemble-Enhance denoiser + LCFM enhancer + UnivNet vocoder from the repo's DeepSpeed <c>mp_rank_00_model_states.pt</c> (~700 MB, downloaded on first use), with a strict zero-missing/ zero-unexpected key check inside <c>LoadWeights</c>.</summary>
    internal static async Task<EnhanceRunner> LoadEnhanceAsync(CancellationToken cancel)
    {
        string path = await AudioModelCache.GetAsync(EnhanceRepo, EnhanceCheckpointFile, category: "fx", ct: cancel).ConfigureAwait(false);
        (IReadOnlyDictionary<string, Tensor> weights, IDisposable loader) = DeepSpeedCheckpointConverter.Load(path);
        ResembleEnhancePipeline pipeline = new ResembleEnhancePipeline(ResembleEnhanceConfig.Default, withDenoiserAndVocoder: true);
        pipeline.LoadWeights(weights);
        Logs.Info($"[Audio][Resemble-Enhance] Loaded {EnhanceRepo} (denoiser + LCFM + UnivNet, {weights.Count} tensors, 44.1 kHz).");
        return new EnhanceRunner(pipeline, [loader]);
    }

    /// <summary>Selects the architecture config from the model name: the 6-stem variant adds guitar+piano and MUST use the 6-source config or the final decoder shape mismatches the weights on load.</summary>
    private static HtDemucsConfig ConfigFor(string? modelName)
    {
        string value = (modelName ?? string.Empty).Trim().ToLowerInvariant();
        if (value.Contains("6s", StringComparison.Ordinal))
        {
            return HtDemucsConfig.Htdemucs6s;
        }
        return value.Contains("_ft", StringComparison.Ordinal) || value.EndsWith("ft", StringComparison.Ordinal)
            ? HtDemucsConfig.HtdemucsFt : HtDemucsConfig.Htdemucs;
    }
}
