using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Codecs.Oobleck;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Memory;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>YuE2 — lyrics- and style-conditioned song generation with an editable score, up to six minutes of
/// 48 kHz stereo. A Qwen3-geometry LM plans an ABC transcription and emits one semantic codec token per 25 Hz
/// frame; a second stack of the same geometry but its own weights flow-matches 64-channel acoustic latents while
/// attending to the first model's per-layer KV cache; an Oobleck VAE decodes those to audio.
///
/// <para>Despite the name this shares no architecture with <see cref="YueMusicModel">YuE v1</see> — different
/// stack, different tokenizer, different codec, different sample rate. See
/// <c>docs/Research/YUE2_ARCHITECTURE.md</c>.</para>
///
/// <para>The whole model is one file: the Comfy-Org repack carries the AR stack, the acoustic stack, the VAE and
/// even the tokenizer, so there is no subfolder fetching and no side assets. Weights are CC BY-NC 4.0.</para></summary>
internal static class Yue2MusicModel
{
    private const string DefaultRepo = "Comfy-Org/YuE2";
    private const string Bf16File = "checkpoints/yue2_3b_bf16.safetensors";
    private const string Int8ConvRotFile = "checkpoints/yue2_3b_int8_convrot.safetensors";
    private const string Category = "music";

    /// <summary>The checkpoint participates in the cache key so switching files never serves the previous one's
    /// runner.</summary>
    internal static MusicModelDescriptor Descriptor { get; } = new MusicModelDescriptor
    {
        ManagesOwnWeights = true,
        CacheKey = selector => $"yue2|{ResolveFile(selector.Variant)}|{selector.LocalPath ?? ""}",
        LoadAsync = (context, selector, cancel) => LoadAsync(context, ResolveFile(selector.Variant), selector.LocalPath, cancel),
    };

    /// <summary>Maps a variant suffix to one of the two published checkpoint files.</summary>
    /// <remarks>Both builds load through the same container: <c>int8_convrot</c>'s <c>.weight_scale</c> and
    /// <c>.comfy_quant</c> companions fold onto the weights before conversion, the projections stay packed at one byte
    /// per parameter on a backend with an int8 path, and <see cref="QuantizedWeightPolicy"/> widens them on one
    /// without. An unrecognised suffix is the BF16 release.</remarks>
    internal static string ResolveFile(string variant)
    {
        string lower = (variant ?? string.Empty).Trim().ToLowerInvariant();
        bool int8 = lower.Contains("int8", StringComparison.Ordinal) || lower.Contains("convrot", StringComparison.Ordinal);
        return int8 ? Int8ConvRotFile : Bf16File;
    }

    private static async Task<IMusicRunner> LoadAsync(MusicLoadContext context, string file, string? localPath, CancellationToken cancel)
    {
        // A LoRA reaches the runner cache key (MusicLoadContext.CacheSuffix) but nothing here applies it, so a request
        // carrying one would otherwise get a fresh runner that generates exactly the base model's song.
        if (context.Loras is { Entries.Count: > 0 } loras)
        {
            Logs.Warning($"[YuE2] {loras.Entries.Count} LoRA(s) were requested, but YuE2 has no LoRA merge wired — "
                + "the song will be the base model's.");
        }

        // An explicitly placed checkpoint wins; otherwise take the conventional local directory if it is already
        // populated, and only then reach for the hub — the file is 7.8 GB and a second copy helps nobody.
        string path = localPath is { Length: > 0 } && File.Exists(localPath)
            ? localPath
            : LocatePlaced(file, context.LmQuant) ?? await AudioModelCache.GetAsync(DefaultRepo, file, Category, ct: cancel).ConfigureAwait(false);

        Logs.Info($"[YuE2] loading {path}");
        // One container for either format, and the fold that attaches a quantized weight's scale has to run before
        // the converter renames anything — a converter renames `.weight` and has no rule for `.weight_scale`.
        CheckpointSource source = CheckpointSource.Open(path);
        IDisposable held = source;
        Yue2Pipeline pipeline;
        Yue2Weights weights;
        try
        {
            if (!Yue2CheckpointConverter.IsComfyCheckpoint(source.Header.Descriptors))
            {
                throw new NotSupportedException(
                    $"'{path}' is not a YuE2 checkpoint. Expected the Comfy-Org single-file repack "
                    + "(text_encoders. / model.diffusion_model. / vae. prefixes).");
            }
            weights = Yue2CheckpointConverter.Convert(source.Weights);
            held = new CompositeDisposable(weights, source);

            // Both stacks run on the primary backend — YuE2 reads no ShardStages — so any quant it has no packed-weight
            // kernel for widens here rather than failing inside the first GEMM, minutes into a song.
            QuantizedWeightPolicy.PreparedWeights preparedAr =
                QuantizedWeightPolicy.PrepareForBackend(weights.Ar, context.Backend);
            held = new CompositeDisposable(preparedAr, weights, source);
            QuantizedWeightPolicy.PreparedWeights preparedNar =
                QuantizedWeightPolicy.PrepareForBackend(weights.Nar, context.Backend);
            held = new CompositeDisposable(preparedAr, preparedNar, weights, source);

            Yue2Config config = Yue2Config.V1;
            Yue2Tokenizer tokenizer = new Yue2Tokenizer(weights.TokenizerJson);
            Yue2ArLm ar = new Yue2ArLm(config);
            ar.LoadWeights(weights.Ar);
            Yue2AcousticTransformer acoustic = new Yue2AcousticTransformer(config);
            acoustic.LoadWeights(weights.Nar);
            OobleckVae vae = new OobleckVae(OobleckConfig.Yue2);
            vae.LoadWeights(weights.Vae);
            pipeline = new Yue2Pipeline(config, tokenizer, ar, acoustic, vae);
        }
        catch
        {
            held.Dispose();
            throw;
        }

        MusicAudio Synth(IBackend backend, MusicRequest request, CancellationToken ct)
        {
            Yue2Request yue2 = BuildRequest(request);
            Yue2Result result = pipeline.Generate(backend, yue2,
                (stage, done, total) => Logs.Debug($"[YuE2] {stage} {done}/{total}"), ct);
            // A short budget has two causes that read identically from the number alone, and only one of them is
            // the caller's to fix: asking past YuE2's own ceiling, or a prompt and score that ate the context.
            // Telling someone to shorten their lyrics when the ceiling was binding sends them after the wrong thing.
            double ceiling = Math.Min(yue2.MaxDurationSeconds, Yue2Protocol.MaxDurationSeconds);
            if (yue2.MaxDurationSeconds > Yue2Protocol.MaxDurationSeconds + 0.001)
            {
                Logs.Warning($"[YuE2] {yue2.MaxDurationSeconds:F1}s is past YuE2's {Yue2Protocol.MaxDurationSeconds:F1}s "
                    + $"ceiling; the song was budgeted for {result.BudgetSeconds:F1}s.");
            }
            else if (result.BudgetSeconds < ceiling - 0.001)
            {
                Logs.Warning($"[YuE2] the prompt and score left room for only {result.BudgetSeconds:F1}s of the "
                    + $"{ceiling:F1}s requested; shorten the lyrics or the score for a longer song.");
            }
            if (result.SemanticTruncated)
            {
                Logs.Warning("[YuE2] the song reached its token budget before ending naturally; raise the duration for a complete take.");
            }
            // A caller asking for a duration the token budget cannot cover gets a song that stops mid-phrase, so both
            // truncation flags travel with the audio rather than only reaching a server-side log.
            Dictionary<string, string> meta = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["truncated"] = result.SemanticTruncated ? "true" : "false",
                ["abcTruncated"] = result.AbcTruncated ? "true" : "false",
                // What the context actually granted. Below the requested duration means the prompt ate the budget,
                // which is otherwise indistinguishable from the model simply choosing to end early.
                ["budgetSeconds"] = result.BudgetSeconds.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
            };
            if (result.Abc is { Length: > 0 } score) meta["abc"] = score;
            return MusicAudio.Stereo(result.Left, result.Right) with { Meta = meta };
        }

        // Planning is the whole song's first pass and costs seconds where the render costs minutes, so it is worth
        // asking for alone: the score comes back editable and goes in again through MusicRequest.Yue2Abc.
        ScorePlanResult Plan(IBackend backend, MusicRequest request, CancellationToken ct)
        {
            Yue2Request yue2 = BuildRequest(request);
            (string abc, int[] ids, bool truncated) = pipeline.PlanScore(backend, yue2,
                (done, total) => Logs.Debug($"[YuE2] plan {done}/{total}"), ct);
            // Budget the score we just wrote, not the empty one the request came in with.
            Yue2Budget budget = pipeline.BudgetFor(yue2 with { Abc = abc });
            if (truncated)
            {
                Logs.Warning("[YuE2] the score reached its token budget before ending naturally.");
            }
            return new ScorePlanResult
            {
                Abc = abc,
                Truncated = truncated,
                ScoreTokens = ids.Length,
                PrefixTokens = budget.PrefixTokens,
                BudgetTokens = budget.BudgetTokens,
                BudgetSeconds = budget.BudgetSeconds,
            };
        }

        ScorePlanResult Budget(MusicRequest request)
        {
            Yue2Request yue2 = BuildRequest(request);
            Yue2Budget budget = pipeline.BudgetFor(yue2);
            return new ScorePlanResult
            {
                Abc = yue2.Abc,
                ScoreTokens = budget.ScoreTokens,
                PrefixTokens = budget.PrefixTokens,
                BudgetTokens = budget.BudgetTokens,
                BudgetSeconds = budget.BudgetSeconds,
            };
        }

        return new MusicRunner(Yue2Config.V1.SampleRate, Synth, pipeline, held)
        {
            Planner = Plan,
            Budgeter = Budget,
        };
    }

    /// <summary>The conventional user-placed location, <c>{models}/audio/music/yue2/*.safetensors</c> or <c>*.gguf</c>.</summary>
    /// <remarks>A GGUF build is taken only when <paramref name="quant"/> asks for one and a placed file names that
    /// precision — nothing is quantized at load, and the hub ships no GGUF, so this never becomes a download. Among the
    /// rest the requested variant's own filename wins and anything else is taken in sorted order: enumeration order is
    /// the filesystem's, and picking a different checkpoint run to run is indistinguishable from the model drifting.</remarks>
    private static string? LocatePlaced(string file, AudioLmQuant quant)
    {
        string directory = AudioModelRoot.WeightsDirectory(Category, "yue2");
        if (!Directory.Exists(directory)) return null;
        List<string> candidates = [.. Directory.EnumerateFiles(directory, "*", SearchOption.TopDirectoryOnly)
            .Where(f => f.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
                || f.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            .OrderBy(f => f, StringComparer.Ordinal)];
        string? quantTag = quant switch
        {
            AudioLmQuant.Q4K => "q4_k",
            AudioLmQuant.Q8 => "q8_0",
            _ => null,
        };
        if (quantTag is not null)
        {
            string? gguf = candidates.FirstOrDefault(f => f.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
                && Path.GetFileName(f).Contains(quantTag, StringComparison.OrdinalIgnoreCase));
            if (gguf is not null) return gguf;
        }
        string preferred = Path.GetFileName(file);
        return candidates.FirstOrDefault(f => Path.GetFileName(f).Equals(preferred, StringComparison.OrdinalIgnoreCase))
            ?? candidates.FirstOrDefault(f => f.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>Maps the engine's generic music request onto YuE2's own knobs.</summary>
    /// <remarks>Following the house convention, <see cref="MusicRequest.Genre"/> carries style tags and
    /// <see cref="MusicRequest.Prompt"/> carries lyrics. The two autoregressive passes sample very differently, so
    /// the shared <c>Temperature</c>/<c>TopK</c>/<c>TopP</c>/<c>RepetitionPenalty</c> knobs are applied to the
    /// semantic pass only — the near-greedy score planner keeps its release preset unless the YuE2-specific
    /// <c>Yue2Abc*</c> fields override it.</remarks>
    private static Yue2Request BuildRequest(MusicRequest request)
    {
        Yue2Sampling semantic = Yue2Sampling.Semantic;
        if (request.Temperature is { } temperature) semantic = semantic with { Temperature = (float)temperature };
        if (request.TopK is { } topK and > 0) semantic = semantic with { TopK = topK };
        if (request.TopP is { } topP) semantic = semantic with { TopP = (float)topP };
        if (request.RepetitionPenalty is { } penalty) semantic = semantic with { RepetitionPenalty = (float)penalty };
        if (request.Yue2PenaltyWindow is { } window) semantic = semantic with { PenaltyWindow = window };
        if (request.Yue2MinTokens is { } minTokens) semantic = semantic with { MinTokens = minTokens };

        Yue2Sampling abc = Yue2Sampling.Abc;
        if (request.Yue2AbcTemperature is { } abcTemperature) abc = abc with { Temperature = (float)abcTemperature };
        if (request.Yue2AbcTopP is { } abcTopP) abc = abc with { TopP = (float)abcTopP };
        if (request.Yue2AbcTopK is { } abcTopK and > 0) abc = abc with { TopK = abcTopK };
        if (request.Yue2AbcRepetitionPenalty is { } abcPenalty) abc = abc with { RepetitionPenalty = (float)abcPenalty };
        if (request.Yue2AbcMaxTokens is { } abcMaxTokens and > 0) abc = abc with { MaxTokens = abcMaxTokens };
        if (request.Yue2AbcPenaltyWindow is { } abcWindow) abc = abc with { PenaltyWindow = abcWindow };

        return new Yue2Request
        {
            Style = request.Genre,
            Lyrics = request.Prompt,
            Cot = ParseCot(request.Yue2Cot),
            Abc = request.Yue2Abc,
            Seed = request.Seed,
            MaxDurationSeconds = request.Duration > 0 ? request.Duration : Yue2Protocol.DefaultDurationSeconds,
            AbcSampling = abc,
            SemanticSampling = semantic,
            CfgScale = request.CfgScale is { } cfg ? (float)cfg : null,
            OdeSteps = request.InferSteps is { } steps and > 0 ? steps : Yue2Config.V1.OdeSteps,
        };
    }

    private static Yue2Cot ParseCot(string value) => value.Trim().ToLowerInvariant() switch
    {
        "" or "full" => Yue2Cot.Full,
        "melody" => Yue2Cot.Melody,
        "off" or "none" => Yue2Cot.Off,
        _ => throw new NotSupportedException($"YuE2's planning mode must be 'full', 'melody' or 'off'; got '{value}'."),
    };
}
