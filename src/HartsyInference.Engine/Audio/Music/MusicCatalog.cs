using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>The music model registry: catalog id → descriptor, plus the shared runner cache and the local-checkpoint
/// resolution the placed-weights families (ACE-Step, YuE) use.</summary>
internal static class MusicCatalog
{
    private const int T5MaxTokens = 256;

    /// <summary>Resolves a catalog id to its descriptor, or throws naming what is available.</summary>
    internal static MusicModelDescriptor Resolve(string id)
    {
        if (Registry.TryGetValue(id ?? string.Empty, out MusicModelDescriptor? descriptor))
        {
            return descriptor;
        }
        throw new NotSupportedException(
            $"No music model '{id}' is registered. Available: {string.Join(", ", Registry.Keys)} "
            + "(pass a variant as 'id:variant', e.g. 'acestep:turbo').");
    }

    /// <summary>MusicGen (facebook/musicgen-{small,medium,large}) — 32 kHz; size inferred from the checkpoint.</summary>
    internal static MusicModelDescriptor MusicGen { get; } = new MusicModelDescriptor
    {
        ManagesOwnWeights = true,
        CacheKey = selector => ResolveMusicGenRepo(selector.Variant),
        LoadAsync = (_, selector, cancel) => LoadMusicGenFamilyAsync(ResolveMusicGenRepo(selector.Variant), audioGen: false, cancel),
    };

    /// <summary>AudioGen (facebook/audiogen-medium) — sound effects at 16 kHz.</summary>
    internal static MusicModelDescriptor AudioGen { get; } = new MusicModelDescriptor
    {
        ManagesOwnWeights = true,
        CacheKey = _ => "facebook/audiogen-medium",
        LoadAsync = (_, _, cancel) => LoadMusicGenFamilyAsync("facebook/audiogen-medium", audioGen: true, cancel),
    };

    private static Dictionary<string, MusicModelDescriptor>? _registry;

    // Built on first use, not in a static initializer: the descriptor properties would still be null if the map were
    // materialized while this type's static initializers were still running.
    private static Dictionary<string, MusicModelDescriptor> Registry => _registry ??= new(StringComparer.OrdinalIgnoreCase)
    {
        ["musicgen"] = MusicGen,
        ["audiogen"] = AudioGen,
        [AudioWeightsCatalog.AceStepId] = AceStepMusicModel.Descriptor,
        [AudioWeightsCatalog.YueId] = YueMusicModel.Descriptor,
        ["heartmula"] = HeartMulaMusicModel.Descriptor,
        ["stableaudio"] = StableAudioMusicModel.Descriptor,
    };

    /// <summary>Resolves a placed checkpoint for the registry-backed families: an explicit local path wins, then the
    /// registered variant file (or, for folder checkpoints, the variant directory) under the family's weights dir.</summary>
    internal static string ResolveLocalCheckpoint(string familyId, AudioModelSelector selector)
    {
        if (!string.IsNullOrEmpty(selector.LocalPath))
        {
            // Folder-checkpoint families: the generic resolver matches the first weights FILE it finds under
            // the models tree, but these loaders take the variant DIRECTORY (sharded set + sidecars) — hand
            // them the containing folder or the load dies with "checkpoint folder not found: …/model-00001….safetensors".
            return AudioWeightsCatalog.IsFolderCheckpoint(familyId) && File.Exists(selector.LocalPath)
                ? Path.GetDirectoryName(selector.LocalPath)!
                : selector.LocalPath;
        }
        string baseDirectory = AudioModelRoot.WeightsDirectory("music", familyId);
        string variant = AudioWeightsCatalog.NormalizeVariant(familyId, selector.Variant);
        // Folder-checkpoint families (YuE) load a whole variant DIRECTORY; the download set lands in "{variant}/…".
        if (AudioWeightsCatalog.IsFolderCheckpoint(familyId))
        {
            return string.IsNullOrEmpty(variant) ? baseDirectory : Path.Combine(baseDirectory, variant);
        }
        ModelAsset? primary = AudioWeightsCatalog.Primary(familyId, variant);
        if (primary is not null)
        {
            return ModelDownloader.TargetPath(primary);
        }
        return string.IsNullOrEmpty(variant) ? baseDirectory : Path.Combine(baseDirectory, variant);
    }

    private static string ResolveMusicGenRepo(string variant)
    {
        string id = (variant ?? string.Empty).Trim();
        if (id.Contains('/', StringComparison.Ordinal))
        {
            return id;
        }
        if (id.Contains("large", StringComparison.OrdinalIgnoreCase))
        {
            return "facebook/musicgen-large";
        }
        return id.Contains("medium", StringComparison.OrdinalIgnoreCase) ? "facebook/musicgen-medium" : "facebook/musicgen-small";
    }

    /// <summary>Loads a MusicGen-family checkpoint (decoder + EnCodec + T5 conditioner).</summary>
    private static async Task<IMusicRunner> LoadMusicGenFamilyAsync(string repo, bool audioGen, CancellationToken cancel)
    {
        Dictionary<string, Tensor> decoderWeights;
        Dictionary<string, Tensor> codecWeights;
        Dictionary<string, Tensor> textWeights;
        IDisposable decoderLoader;
        IDisposable codecLoader;
        IDisposable textLoader;
        // AudioGen — and MusicGen large, whose combined HF file is sharded — load from the AudioCraft single-file
        // pickles instead: decoder = state_dict.bin, EnCodec = compression_state_dict.bin, T5 = a standalone repo.
        bool audioCraft = audioGen || repo.Contains("large", StringComparison.OrdinalIgnoreCase);
        if (audioCraft)
        {
            string decoderPath = await AudioModelCache.GetAsync(repo, "state_dict.bin", category: "music", ct: cancel).ConfigureAwait(false);
            string codecPath = await AudioModelCache.GetAsync(repo, "compression_state_dict.bin", category: "music", ct: cancel).ConfigureAwait(false);
            // AudioGen conditions on t5-LARGE (enc_to_dec_proj input = 1024); MusicGen-large uses t5-base.
            string t5Repo = audioGen ? "google-t5/t5-large" : "google-t5/t5-base";
            string t5Path = await AudioModelCache.GetAsync(t5Repo, "pytorch_model.bin", category: "music", ct: cancel).ConfigureAwait(false);
            (decoderWeights, decoderLoader) = MusicGenCheckpointConverter.LoadDecoderAny(decoderPath, castToF32: true);
            (codecWeights, codecLoader) = MusicGenCheckpointConverter.LoadEnCodecAny(codecPath, castToF32: true);
            (textWeights, textLoader) = MusicGenCheckpointConverter.LoadTextEncoderAny(t5Path, castToF32: true);
        }
        else
        {
            // MusicGen small/medium ship one combined file: small = model.safetensors, medium = pytorch_model.bin.
            string path = await ResolveMusicGenCombinedAsync(repo, cancel).ConfigureAwait(false);
            (decoderWeights, decoderLoader) = MusicGenCheckpointConverter.LoadDecoderAny(path, castToF32: true);
            (codecWeights, codecLoader) = MusicGenCheckpointConverter.LoadEnCodecAny(path, castToF32: true);
            (textWeights, textLoader) = MusicGenCheckpointConverter.LoadTextEncoderAny(path, castToF32: true);
        }

        MusicGenConfig config = audioGen ? MusicGenConfig.AudioGen : InferMusicGenSize(decoderWeights, repo);
        MusicGenDecoder decoder = new MusicGenDecoder(config);
        decoder.LoadWeights(decoderWeights, prefix: "model.decoder");
        EnCodec codec = new EnCodec(audioGen ? EnCodecConfig.EnCodec16kHz : EnCodecConfig.EnCodec32kHz);
        codec.LoadWeights(codecWeights);
        // AudioGen conditions on t5-large (dim 1024); MusicGen on t5-base.
        T5TextEncoderConfig textConfig = audioGen
            ? new T5TextEncoderConfig
            {
                DModel = 1024,
                DFf = 4096,
                DKv = 64,
                NumHeads = 16,
                NumLayers = 24,
                VocabSize = 32128,
                GatedFeedForward = false,
                UseReluActivation = true,
                AttentionScale = 1.0f,
            }
            : T5TextEncoderConfig.T5Base;
        T5TextEncoder textEncoder = new T5TextEncoder(textConfig);
        textEncoder.LoadWeights(textWeights);
        T5Tokenizer tokenizer = new T5Tokenizer(maxLength: T5MaxTokens);

        MusicGenPipeline pipeline = new MusicGenPipeline(config, decoder, codec);
        Logs.Info($"[Audio][MusicGen] Loaded {repo} ({config.CodecSampleRate} Hz).");

        MusicAudio Synth(IBackend backend, MusicRequest request, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            // Real-length ids + </s>, NO padding: the MusicGen decoder cross-attention has no encoder mask, so padded
            // T5 rows would swamp the text conditioning. HF appends EOS and masks pad — real-length+EOS is equivalent.
            IReadOnlyList<int> rawIds = tokenizer.EncodeRaw(request.Prompt);
            int[] tokens = new int[rawIds.Count + 1];
            for (int i = 0; i < rawIds.Count; i++)
            {
                tokens[i] = rawIds[i];
            }
            tokens[rawIds.Count] = T5Tokenizer.EosTokenId;
            Tensor textStates = textEncoder.Encode(backend, [tokens], [T5Tokenizer.CreateAttentionMask(tokens)]);
            backend.Sync();
            backend.FreeWeights(textEncoder.EnumerateWeights());
            try
            {
                // MusicGen/AudioGen are trained on windows of at most 30 s.
                float[] samples = pipeline.Synthesize(backend, textStates, seconds: (float)Math.Clamp(request.Duration, 1d, 30d), seed: request.Seed);
                return MusicAudio.Mono(samples);
            }
            finally
            {
                textStates.Dispose();
            }
        }

        return new MusicRunner(config.CodecSampleRate, Synth, textEncoder, tokenizer, decoderLoader, codecLoader, textLoader);
    }

    /// <summary>The MusicGen combined checkpoint: small ships model.safetensors; medium ships pytorch_model.bin.</summary>
    private static async Task<string> ResolveMusicGenCombinedAsync(string repo, CancellationToken cancel)
    {
        try
        {
            return await AudioModelCache.GetAsync(repo, "model.safetensors", category: "music", ct: cancel).ConfigureAwait(false);
        }
        catch (FileNotFoundException ex)
        {
            Logs.Debug($"[Audio][MusicGen] '{repo}' has no model.safetensors ({ex.Message}); using pytorch_model.bin.");
            return await AudioModelCache.GetAsync(repo, "pytorch_model.bin", category: "music", ct: cancel).ConfigureAwait(false);
        }
    }

    /// <summary>Infers Small/Medium/Large from the decoder's final layer-norm width (1024/1536/2048).</summary>
    private static MusicGenConfig InferMusicGenSize(Dictionary<string, Tensor> decoderWeights, string repo)
    {
        int hidden = decoderWeights.TryGetValue("model.decoder.layer_norm.weight", out Tensor? finalNorm) ? (int)finalNorm.Shape[0] : 0;
        return hidden switch
        {
            1024 => MusicGenConfig.Small,
            1536 => MusicGenConfig.Medium,
            2048 => MusicGenConfig.Large,
            _ => throw new InvalidOperationException(
                $"MusicGen checkpoint '{repo}' has unrecognized decoder width {hidden} (expected 1024/1536/2048)."),
        };
    }
}
