using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Codecs.Oobleck;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
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
    private const string Category = "music";

    /// <summary>The checkpoint participates in the cache key so switching files never serves the previous one's
    /// runner.</summary>
    internal static MusicModelDescriptor Descriptor { get; } = new MusicModelDescriptor
    {
        ManagesOwnWeights = true,
        CacheKey = selector => $"yue2|{ResolveFile(selector.Variant)}|{selector.LocalPath ?? ""}",
        LoadAsync = (context, selector, cancel) => LoadAsync(context, ResolveFile(selector.Variant), selector.LocalPath, cancel),
    };

    /// <summary>Maps a variant suffix to a checkpoint file. Only the BF16 build is wired today; the
    /// <c>int8_convrot</c> repack needs the per-layer quant reader and is refused by name rather than silently
    /// falling back to a different precision than the caller asked for.</summary>
    private static string ResolveFile(string variant)
    {
        string lower = (variant ?? string.Empty).Trim().ToLowerInvariant();
        if (lower.Contains("int8", StringComparison.Ordinal) || lower.Contains("convrot", StringComparison.Ordinal))
        {
            throw new NotSupportedException(
                "YuE2's int8_convrot checkpoint is not wired yet; use the default (BF16) variant.");
        }
        return Bf16File;
    }

    private static async Task<IMusicRunner> LoadAsync(MusicLoadContext context, string file, string? localPath, CancellationToken cancel)
    {
        // An explicitly placed checkpoint wins; otherwise take the conventional local directory if it is already
        // populated, and only then reach for the hub — the file is 7.8 GB and a second copy helps nobody.
        string path = localPath is { Length: > 0 } && File.Exists(localPath)
            ? localPath
            : LocatePlaced() ?? await AudioModelCache.GetAsync(DefaultRepo, file, Category, ct: cancel).ConfigureAwait(false);

        Logs.Info($"[YuE2] loading {path}");
        SafeTensorsLoader loader = new SafeTensorsLoader();
        Yue2Pipeline pipeline;
        Yue2Weights weights;
        try
        {
            loader.Load(path);
            if (!Yue2CheckpointConverter.IsComfyCheckpoint(loader.Descriptors))
            {
                throw new NotSupportedException(
                    $"'{path}' is not a YuE2 checkpoint. Expected the Comfy-Org single-file repack "
                    + "(text_encoders. / model.diffusion_model. / vae. prefixes).");
            }
            weights = Yue2CheckpointConverter.Convert(loader.GetAllTensors());

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
            loader.Dispose();
            throw;
        }

        MusicAudio Synth(IBackend backend, MusicRequest request, CancellationToken ct)
        {
            Yue2Request yue2 = BuildRequest(request);
            Yue2Result result = pipeline.Generate(backend, yue2,
                (stage, done, total) => Logs.Debug($"[YuE2] {stage} {done}/{total}"), ct);
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
            };
            if (result.Abc is { Length: > 0 } score) meta["abc"] = score;
            return MusicAudio.Stereo(result.Left, result.Right) with { Meta = meta };
        }

        return new MusicRunner(Yue2Config.V1.SampleRate, Synth, pipeline, weights, loader);
    }

    /// <summary>The conventional user-placed location, <c>{models}/audio/music/yue2/*.safetensors</c>.</summary>
    private static string? LocatePlaced()
    {
        string directory = AudioModelRoot.WeightsDirectory(Category, "yue2");
        if (!Directory.Exists(directory)) return null;
        // The quant repack is not wired yet, so it is never a candidate. Among the rest the canonical release name
        // wins and anything else is taken in sorted order — enumeration order is the filesystem's, and picking a
        // different checkpoint run to run would be indistinguishable from the model drifting.
        List<string> candidates = [.. Directory.EnumerateFiles(directory, "*.safetensors", SearchOption.TopDirectoryOnly)
            .Where(f => !Path.GetFileName(f).Contains("int8", StringComparison.OrdinalIgnoreCase))
            .OrderBy(f => f, StringComparer.Ordinal)];
        string preferred = Path.GetFileName(Bf16File);
        return candidates.FirstOrDefault(f => Path.GetFileName(f).Equals(preferred, StringComparison.OrdinalIgnoreCase))
            ?? candidates.FirstOrDefault();
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

        return new Yue2Request
        {
            Style = request.Genre,
            Lyrics = request.Prompt,
            Cot = ParseCot(request.Yue2Cot),
            Abc = request.Yue2Abc,
            Seed = request.Seed,
            MaxDurationSeconds = request.Duration > 0 ? request.Duration : Yue2Protocol.MaxDurationSeconds,
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
