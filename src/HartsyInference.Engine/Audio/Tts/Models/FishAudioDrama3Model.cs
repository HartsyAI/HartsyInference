using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Collections;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Placement;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>Fish Audio Drama 3 (<c>fishaudio/drama-3</c>, block-FP8 experts; <c>fishaudio/drama-3-int4</c>, GPTQ int4): a 35B-A3B
/// Qwen3.5-MoE text-to-speech model with a codebook tower over 10 codebooks, decoded by S2's ModifiedDAC codec to 44.1 kHz mono.
/// Inline <c>[bracket cues]</c> direct the delivery and <c>&lt;|speaker:N|&gt;</c> tags switch speakers. Voices clone from one or
/// more references (each a transcript with its clip), and a take is revised by sending its returned turns back as history with a
/// new cue. The weights ship under the Fish Audio Community License v1.0: commercial use below USD 500k annual revenue, no
/// Hosted Services at any revenue, and no imitation of a real person's voice without consent.
///
/// <para>The variant picks the build and its precision, joined by <c>+</c>: <c>int4</c> selects the GPTQ release, an
/// <c>org/repo</c> any other build of the same layout (the unquantized original included), and <c>bf16</c>, <c>q8</c>, <c>q6</c>,
/// <c>q5</c> or <c>q4</c> re-quantizes the weights once into a disk cache. Without a precision each build loads at its own:
/// FP8 experts as Q8_0, GPTQ as Q4_0 (both near or exactly lossless), plain weights as stored.</para>
///
/// <para>On a GPU the model is placed by <see cref="TextPlacementPlanner"/>: everything resident when it fits, otherwise the dense
/// weights resident and the routed experts behind a CUDA expert cache, the misses running on the CPU.</para></summary>
internal static class FishAudioDrama3Model
{
    internal const string Repo = "fishaudio/drama-3";
    internal const string Int4Repo = "fishaudio/drama-3-int4";

    /// <summary>Upstream decodes with S2 Pro's codec when a Drama 3 repo does not carry its own.</summary>
    internal const string CodecRepo = "fishaudio/s2-pro";

    /// <summary>The catalog ids this model registers under; a bare id reaches the descriptor as its own variant.</summary>
    internal static readonly string[] CatalogIds = ["fishaudio-drama3", "drama3"];

    private const string Category = "tts";
    private const string ConfigFile = "config.json";
    private const string TokenizerFile = "tokenizer.json";
    private const string CodecFile = "codec.pth";
    private const string LogTag = "[Audio][Drama3]";

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = variant => FishAudioDrama3Variant.Parse(variant).Repo,
        ResolveCacheTag = variant => FishAudioDrama3Variant.Parse(variant).CacheTag,
        ResolveFiles = (variant, cancel) => ResolveFilesAsync(FishAudioDrama3Variant.Parse(variant).Repo, cancel),
        LoadAsync = (context, variant, cancel) => LoadAsync(context, FishAudioDrama3Variant.Parse(variant), cancel),
    };

    /// <summary>The files a build pulls, companions first and the weights last: the config, the tokenizer, the codec (from S2 Pro
    /// when the build has none) and the safetensors shards.</summary>
    private static async Task<IReadOnlyList<AudioModelFile>> ResolveFilesAsync(string repo, CancellationToken cancel)
    {
        bool ownCodec = await AudioModelCache.ExistsAsync(repo, CodecFile, Category, ct: cancel).ConfigureAwait(false);
        List<AudioModelFile> files = [new(ConfigFile), new(TokenizerFile), new(CodecFile, Repo: ownCodec ? null : CodecRepo)];
        files.AddRange(await AudioCheckpoints.ResolveCheckpointFilesAsync(repo, Category, cancel).ConfigureAwait(false));
        return files;
    }

    private static async Task<ITtsRunner> LoadAsync(TtsLoadContext context, FishAudioDrama3Variant variant, CancellationToken cancel)
    {
        IBackend backend = context.Backend;
        IReadOnlyList<AudioModelFile> files = await ResolveFilesAsync(variant.Repo, cancel).ConfigureAwait(false);
        IReadOnlyDictionary<string, string> paths =
            await AudioModelCache.FetchAllAsync(variant.Repo, files, Category, ct: cancel).ConfigureAwait(false);
        FishAudioDrama3Config config = FishAudioDrama3Config.FromJson(await File.ReadAllTextAsync(paths[ConfigFile], cancel).ConfigureAwait(false));
        GgufTokenizer tokenizer;
        using (FileStream json = File.OpenRead(paths[TokenizerFile]))
        {
            tokenizer = HfTokenizerJson.LoadByteLevelBpe(json);
        }
        ValidateTokenizer(tokenizer, config);
        string[] shards = [.. files.Where(static f => f.Name.EndsWith(".safetensors", StringComparison.Ordinal)).Select(f => paths[f.Name])];

        HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles(shards, config.GptqGroupSize, config.Fp8BlockSize);
        IReadOnlyDictionary<string, Tensor> model;
        IDisposable weightsOwner;
        try
        {
            RefuseUnreadableWeights(backend, checkpoint, variant.Precision);
            // The owner is the checkpoint itself when its weights are used as stored, else the cache, the checkpoint then closed.
            (model, weightsOwner) = FishAudioDrama3Weights.Load(checkpoint, variant.Precision, CachePath(variant));
        }
        catch
        {
            checkpoint.Dispose();
            throw;
        }

        // Disposed in list order: the pipeline, the expert offload, then the mappings their weights live in.
        List<IDisposable> owned = [weightsOwner];
        try
        {
            AnyFormatCheckpointLoader codecLoader = new();
            owned.Add(codecLoader);
            codecLoader.Load(paths[CodecFile]);

            FishAudioDrama3Pipeline pipeline = new(config, ModifiedDacConfig.S2, text => tokenizer.Encode(text, addSpecial: true));
            owned.Insert(0, pipeline);
            pipeline.LoadWeights(model, codecLoader.GetAllTensors(),
                FishAudioDrama3Config.MaxPromptRows + FishAudioDrama3Config.DefaultMaxNewFrames);
            MoeExpertOffload? offload = Place(backend, pipeline.Model!);
            if (offload is not null) owned.Insert(1, offload);
            Logs.Info($"{LogTag} Loaded {variant.Repo} at {variant.Precision} (Qwen3.5-MoE + codebook tower + ModifiedDAC, 44.1 kHz).");

            Session session = new(pipeline);
            return new StreamingTtsRunner(pipeline.SampleRate, session.Synthesize, session.SynthesizeStream, [.. owned]);
        }
        catch
        {
            foreach (IDisposable disposable in owned) disposable.Dispose();
            throw;
        }
    }

    /// <summary>Where a re-quantized build is cached: beside the other audio quant caches, never inside the repo folder, so the
    /// weight footprint of the repo does not count it.</summary>
    private static string CachePath(FishAudioDrama3Variant variant) => Path.Combine(AudioModelCache.CacheRoot, Category, "_gguf-cache",
        $"{variant.Repo.Replace('/', '_')}-{variant.Precision.ToString().ToLowerInvariant()}-v{FishAudioDrama3Weights.CacheVersion}.gguf");

    /// <summary>The prompt is built from the tokenizer's ids and the model samples from the config's semantic band: a tokenizer from
    /// another build would put every code on the wrong token.</summary>
    private static void ValidateTokenizer(GgufTokenizer tokenizer, FishAudioDrama3Config config)
    {
        int last = config.SemanticEndId - config.SemanticStartId;
        (string Literal, int Expected)[] anchors =
        [
            ("<|semantic:0|>", config.SemanticStartId),
            ($"<|semantic:{last}|>", config.SemanticEndId),
            ("<|im_end|>", config.EosId),
        ];
        foreach ((string literal, int expected) in anchors)
        {
            if (tokenizer.SpecialId(literal) != expected)
            {
                throw new InvalidDataException($"The Drama 3 tokenizer maps '{literal}' to {tokenizer.SpecialId(literal)?.ToString() ?? "nothing"}; "
                    + $"its config expects {expected}. The tokenizer and the weights come from different builds.");
            }
        }
    }

    /// <summary>Refuses a precision whose quantized weights <paramref name="backend"/> cannot read, before a cache is written for it.</summary>
    private static void RefuseUnreadableWeights(IBackend backend, HfQuantCheckpoint checkpoint, FishAudioDrama3Precision precision)
    {
        foreach (DType dtype in checkpoint.Weights.Values.Select(w => FishAudioDrama3Weights.TargetDType(w, precision)).Distinct())
        {
            if (dtype.IsQuantized && !backend.Capabilities.SupportsQuantizedType(dtype))
            {
                throw new NotSupportedException($"Fish Audio Drama 3 at {precision} keeps weights in {dtype.Name}, which "
                    + $"{backend.Capabilities.Name} cannot read. Use the 'bf16' variant on this device.");
            }
        }
    }

    /// <summary>Makes the model resident on <paramref name="backend"/>: everything when it fits, else the dense weights plus a CUDA
    /// expert cache. Returns the offload when one was attached; null when everything is resident or the backend has no device memory.</summary>
    /// <exception cref="HartsyInferenceException">Neither placement fits.</exception>
    private static MoeExpertOffload? Place(IBackend backend, FishAudioDrama3Lm lm)
    {
        (long free, long total) = backend.GetVramInfo();
        if (total <= 0)
        {
            return null;
        }
        List<Tensor> experts = [.. lm.Backbone.EnumerateExpertWeights()];
        long expertBytes = experts.Sum(static t => t.DType.ComputeByteCount(t.ElementCount));
        long denseBytes = lm.EnumerateWeightGroups(includeExperts: false).SelectMany(static g => g)
            .Sum(static t => t.DType.ComputeByteCount(t.ElementCount));
        string? hostReason = TextPlacementDemandReader.ExpertsHostReason(experts.Select(static t => t.DType),
            lm.Backbone.ExpertHiddenSize, lm.Backbone.ExpertIntermediateSize);
        // Prefill runs in chunks of PrefillChunk rows, whose activations the planner's reserve covers, so none are charged per row.
        TextPlacementDemand demand = new(denseBytes, expertBytes, lm.Config.Backbone.NumLayers, lm.KvBytesPerRow, lm.Capacity)
        {
            ExpertsHostRunnable = hostReason is null,
            ExpertsHostReason = hostReason,
        };
        TextPlacement plan = TextPlacementPlanner.Plan(demand, [new TextPlacementDevice(backend.Capabilities.Name, free)],
            HostMemoryInfo.AvailableBytes(), TextPlacementMode.Auto);
        if (!plan.Feasible)
        {
            throw new HartsyInferenceException($"Fish Audio Drama 3 does not fit this device. {plan.Reason}");
        }
        Logs.Info($"{LogTag} {plan.Reason}");
        if (plan.Mode != TextPlacementMode.Offload)
        {
            backend.PreloadWeightGroups(lm.EnumerateWeightGroups());
            return null;
        }
        return ExpertOffloadSetup.Attach(backend, lm.Backbone, lm.EnumerateWeightGroups(includeExperts: false), plan.ExpertBudgetBytes, LogTag);
    }

    /// <summary>Turns a request turn into the model's own: its codes as given, or its audio encoded by the codec.</summary>
    private static FishAudioDrama3Turn ToModelTurn(FishAudioDrama3Pipeline pipeline, IBackend backend, SpeechTurn turn, string role)
    {
        if (turn.Codes is { Count: > 0 } codes)
        {
            return new FishAudioDrama3Turn(turn.Text, Grid2D.FromRows(codes));
        }
        if (turn.Audio is { Data.Length: > 0 } audio)
        {
            return new FishAudioDrama3Turn(turn.Text, pipeline.EncodeReference(backend, AudioClipCodec.DecodeMono(audio, pipeline.SampleRate)));
        }
        throw new ArgumentException($"A Drama 3 {role} needs its codes or its audio.");
    }

    /// <summary>The pipeline request for <paramref name="job"/>: the single reference first, then the listed ones, then the history.</summary>
    private static FishAudioDrama3Pipeline.Request BuildRequest(FishAudioDrama3Pipeline pipeline, IBackend backend, TtsJob job, CancellationToken cancel)
    {
        List<SpeechTurn> references = [];
        if (job.Reference is { Data.Length: > 0 })
        {
            references.Add(new SpeechTurn { Text = job.RefText, Audio = job.Reference });
        }
        references.AddRange(job.References);
        foreach (SpeechTurn reference in references)
        {
            if (string.IsNullOrWhiteSpace(reference.Text))
            {
                throw new ArgumentException("Fish Audio Drama 3 voice cloning needs the exact transcript of every reference clip.");
            }
        }
        return new FishAudioDrama3Pipeline.Request
        {
            Text = job.Text,
            References = [.. references.Select(r => ToModelTurn(pipeline, backend, r, "reference"))],
            History = [.. job.History.Select(h => ToModelTurn(pipeline, backend, h, "history turn"))],
            Temperature = job.Temperature is { } t ? (float)t : null,
            TopP = job.TopP is { } p ? (float)p : null,
            TopK = job.TopK,
            MaxFrames = job.MaxTokens,
            Seed = job.Seed,
            Cancel = cancel,
        };
    }

    /// <summary>The loaded pipeline behind both runner entry points; the runner owns its disposal.</summary>
    private sealed class Session(FishAudioDrama3Pipeline pipeline)
    {
        public float[] Synthesize(IBackend backend, TtsJob job) => Run(backend, job, job.Cancel, onAudio: null);

        /// <summary>Streams each chunk's audio while it generates (see <see cref="FishAudioDrama3Pipeline.Synthesize"/>).</summary>
        public IAsyncEnumerable<AudioChunk> SynthesizeStream(IBackend backend, TtsJob job, CancellationToken cancel) =>
            AudioStreamer.FromProducer(emit => Run(backend, job, cancel, emit), pipeline.SampleRate, cancel);

        private float[] Run(IBackend backend, TtsJob job, CancellationToken cancel, Action<float[]>? onAudio)
        {
            FishAudioDrama3Pipeline.Result result = pipeline.Synthesize(backend, BuildRequest(pipeline, backend, job, cancel), onAudio);
            job.TurnSink?.Invoke([.. result.Turns.Select(static t => new SpeechTurn { Text = t.Text, Codes = Grid2D.ToRows(t.Codes) })]);
            return result.Audio;
        }
    }
}

/// <summary>A Drama 3 variant: the build to load and the precision to load it at.</summary>
/// <param name="Repo">The Hugging Face repo of the build.</param>
/// <param name="Precision">The precision its weights load at.</param>
internal readonly record struct FishAudioDrama3Variant(string Repo, FishAudioDrama3Precision Precision)
{
    /// <summary>Tells loads of one repo apart in the runner cache: the precision, or empty for the build's own.</summary>
    internal string CacheTag => Precision == FishAudioDrama3Precision.Native ? "" : Precision.ToString().ToLowerInvariant();

    /// <summary>Parses <c>+</c>- or <c>,</c>-joined tokens: <c>int4</c> (or <c>gptq</c>) for the GPTQ build, <c>fp8</c> for the
    /// default one, an <c>org/repo</c> for any other, and a precision (<see cref="FishAudioDrama3Weights.ParsePrecision"/>). A
    /// catalog id is the bare request and selects the defaults.</summary>
    /// <exception cref="ArgumentException">A token is none of these, or two tokens pick different builds or precisions.</exception>
    internal static FishAudioDrama3Variant Parse(string? variant)
    {
        string? repo = null;
        FishAudioDrama3Precision? precision = null;
        foreach (string token in (variant ?? "").Split(['+', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (FishAudioDrama3Model.CatalogIds.Contains(token, StringComparer.OrdinalIgnoreCase))
            {
                continue;
            }
            string? tokenRepo = token.Contains('/', StringComparison.Ordinal) ? token : token.ToLowerInvariant() switch
            {
                "int4" or "gptq" => FishAudioDrama3Model.Int4Repo,
                "fp8" => FishAudioDrama3Model.Repo,
                _ => null,
            };
            if (tokenRepo is not null)
            {
                repo = repo is null || repo == tokenRepo ? tokenRepo
                    : throw new ArgumentException($"The Drama 3 variant '{variant}' names two builds, {repo} and {tokenRepo}.");
                continue;
            }
            if (!FishAudioDrama3Weights.TryParsePrecision(token, out FishAudioDrama3Precision parsed))
            {
                throw new ArgumentException($"'{token}' in the Drama 3 variant '{variant}' is not a build (int4, fp8 or org/repo) "
                    + "or a precision (native, bf16, q8, q6, q5 or q4).");
            }
            precision = precision is null || precision == parsed ? parsed
                : throw new ArgumentException($"The Drama 3 variant '{variant}' names two precisions, {precision} and {parsed}.");
        }
        return new FishAudioDrama3Variant(repo ?? FishAudioDrama3Model.Repo, precision ?? FishAudioDrama3Precision.Native);
    }
}
