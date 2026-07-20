using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Kyutai;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>The speech-to-text model registry: catalog id → descriptor, plus the shared runner cache.</summary>
internal static class SttCatalog
{
    /// <summary>Resident transcription pipelines, keyed by resolved repo.</summary>
    internal static readonly AudioRunnerCache<ISttRunner> Cache = new AudioRunnerCache<ISttRunner>();

    private static Dictionary<string, SttModelDescriptor>? _registry;

    // Built on first use, not in a static initializer: the descriptor properties below would still be null if the
    // map were materialized while this type's static initializers were still running.
    private static Dictionary<string, SttModelDescriptor> Registry => _registry ??= new(StringComparer.OrdinalIgnoreCase)
    {
        ["whisper"] = Whisper,
        ["distilwhisper"] = DistilWhisper,
        ["moonshine"] = Moonshine,
        ["kyutaistt"] = Kyutai,
        ["whisperstreaming"] = WhisperStreaming,
    };

    /// <summary>Resolves a catalog id to its descriptor, or throws naming what is available.</summary>
    internal static SttModelDescriptor Resolve(string id)
    {
        if (Registry.TryGetValue(id ?? string.Empty, out SttModelDescriptor? descriptor))
        {
            return descriptor;
        }
        throw new NotSupportedException(
            $"No transcription model '{id}' is registered. Available: {string.Join(", ", Registry.Keys)} "
            + "(pass a variant as 'id:variant', e.g. 'whisper:large-v3').");
    }

    /// <summary>OpenAI Whisper. Honors the request's language and translate task.</summary>
    internal static SttModelDescriptor Whisper { get; } = new SttModelDescriptor
    {
        ResolveRepo = ResolveWhisperRepo,
        LoadAsync = async (repo, cancel) =>
        {
            WhisperPipeline pipeline = await WhisperPipeline.LoadAsync(repo, ct: cancel).ConfigureAwait(false);
            return new SttRunner((backend, audio, request) =>
                pipeline.TranscribeAudio(backend, audio, 16_000, ToWhisperOptions(request)), pipeline)
            {
                Timed = (backend, audio, request) => ToSegments(pipeline.SegmentAudio(backend, audio, 16_000, ToWhisperOptions(request))),
            };
        },
    };

    /// <summary>distil-whisper — the same pipeline, but its model ids ("large-v3") do not contain "distil", so it
    /// needs its own resolver.</summary>
    internal static SttModelDescriptor DistilWhisper { get; } = new SttModelDescriptor
    {
        ResolveRepo = ResolveDistilWhisperRepo,
        LoadAsync = async (repo, cancel) =>
        {
            WhisperPipeline pipeline = await WhisperPipeline.LoadAsync(repo, ct: cancel).ConfigureAwait(false);
            return new SttRunner((backend, audio, request) =>
                pipeline.TranscribeAudio(backend, audio, 16_000, ToWhisperOptions(request)), pipeline)
            {
                Timed = (backend, audio, request) => ToSegments(pipeline.SegmentAudio(backend, audio, 16_000, ToWhisperOptions(request))),
            };
        },
    };

    /// <summary>Useful Sensors Moonshine (tiny/base). No language/task tokens — those request fields are ignored.</summary>
    internal static SttModelDescriptor Moonshine { get; } = new SttModelDescriptor
    {
        ResolveRepo = ResolveMoonshineRepo,
        LoadAsync = async (repo, cancel) =>
        {
            MoonshinePipeline pipeline = await MoonshinePipeline.LoadAsync(repo, ct: cancel).ConfigureAwait(false);
            return new SttRunner((backend, audio, _) => pipeline.TranscribeAudio(backend, audio, 16_000), pipeline);
        },
    };

    /// <summary>Kyutai delayed-streams STT — Helium LM + Mimi codec → text token ids, decoded by the SentencePiece
    /// tokenizer. Input audio is 24 kHz; the architecture has no language/task tokens.</summary>
    internal static SttModelDescriptor Kyutai { get; } = new SttModelDescriptor
    {
        InputSampleRate = 24_000,
        // The engine loader expects the HF Transformers ("-trfs") key layout.
        ResolveRepo = variant =>
        {
            string id = (variant ?? string.Empty).Trim();
            if (id.Contains('/', StringComparison.Ordinal))
            {
                return id;
            }
            return id.Contains("1b", StringComparison.OrdinalIgnoreCase) ? "kyutai/stt-1b-en_fr-trfs" : "kyutai/stt-2.6b-en-trfs";
        },
        LoadAsync = async (repo, cancel) =>
        {
            (IReadOnlyDictionary<string, Tensor> dict, IDisposable[] loaders) = await AudioCheckpoints.LoadAsync(repo, cancel).ConfigureAwait(false);
            bool small = repo.Contains("1b", StringComparison.OrdinalIgnoreCase);
            // The SentencePiece text model is not shipped in the -trfs repos — fetch it from the original repo.
            (string spmRepo, string spmFile) = small
                ? ("kyutai/stt-1b-en_fr", "tokenizer_en_fr_audio_8000.model")
                : ("kyutai/stt-2.6b-en", "tokenizer_en_audio_4000.model");
            string spm = await AudioModelCache.GetAsync(spmRepo, spmFile, ct: cancel).ConfigureAwait(false);
            KyutaiSttConfig config = small ? KyutaiSttConfig.Stt1B : KyutaiSttConfig.Stt2_6B;
            // The -trfs checkpoint namespaces the Mimi codec under "codec_model."; strip it so the engine keys match.
            Dictionary<string, Tensor> mimi = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            foreach (KeyValuePair<string, Tensor> entry in dict)
            {
                if (entry.Key.StartsWith("codec_model.", StringComparison.Ordinal))
                {
                    mimi[entry.Key["codec_model.".Length..]] = entry.Value;
                }
            }
            KyutaiSttPipeline pipeline = new KyutaiSttPipeline(config);
            pipeline.LoadWeights(dict, mimi);
            KyutaiSttTokenizer tokenizer = new KyutaiSttTokenizer(spm);
            IDisposable?[] keep = [pipeline, tokenizer, .. loaders];
            return new SttRunner((backend, audio, _) => tokenizer.Decode(pipeline.Transcribe(backend, audio)), keep);
        },
    };

    /// <summary>Whisper Streaming — the same Whisper weights driven through the LocalAgreement-2 hypothesis buffer.
    /// For a request/response call the whole clip is pushed and flushed, so the result equals batch Whisper with the
    /// streaming stabilizer; the value is the live partial/confirmed API for future real-time callers.</summary>
    internal static SttModelDescriptor WhisperStreaming { get; } = new SttModelDescriptor
    {
        ResolveRepo = ResolveWhisperRepo,
        LoadAsync = async (repo, cancel) =>
        {
            WhisperPipeline pipeline = await WhisperPipeline.LoadAsync(repo, ct: cancel).ConfigureAwait(false);
            return new SttRunner((backend, audio, request) =>
            {
                using WhisperStreamingPipeline stream = new WhisperStreamingPipeline(pipeline, backend, ToWhisperOptions(request));
                stream.PushAudio(audio, 16_000);
                return stream.Finish();
            }, pipeline)
            {
                // The stabilizer carries no timing of its own, so timestamps come from the underlying batch decode —
                // which the streaming pipeline's own docs describe as the same result for a whole-clip call.
                Timed = (backend, audio, request) => ToSegments(pipeline.SegmentAudio(backend, audio, 16_000, ToWhisperOptions(request))),
            };
        },
    };

    /// <summary>Projects Whisper's native timestamp spans onto the engine-side segment type.</summary>
    private static IReadOnlyList<SttSegment> ToSegments(IReadOnlyList<WhisperSegment> segments)
    {
        List<SttSegment> result = new List<SttSegment>(segments.Count);
        foreach (WhisperSegment segment in segments)
        {
            result.Add(new SttSegment { Text = segment.Text, Start = segment.Start, End = segment.End });
        }
        return result;
    }

    /// <summary>Maps the typed request to the engine's Whisper decode options.</summary>
    private static WhisperOptions ToWhisperOptions(AudioRequest request) => new WhisperOptions
    {
        Language = string.IsNullOrEmpty(request.Language) ? null : request.Language,
        Translate = request.Translate,
    };

    /// <summary>Whisper / distil-whisper variant → HF repo. Full repo ids pass through; otherwise a size token is
    /// matched, else a sensible family default.</summary>
    private static string ResolveWhisperRepo(string variant)
    {
        string id = (variant ?? string.Empty).Trim();
        if (id.Contains('/', StringComparison.Ordinal))
        {
            return id;
        }
        string lower = id.ToLowerInvariant();
        if (lower.Contains("distil", StringComparison.Ordinal))
        {
            return ResolveDistilWhisperRepo(id);
        }
        // "turbo" is large-v3-turbo and has no "large" substring — match it first.
        if (lower.Contains("turbo", StringComparison.Ordinal))
        {
            return "openai/whisper-large-v3-turbo";
        }
        if (lower.Contains("large", StringComparison.Ordinal))
        {
            return lower.Contains("v2", StringComparison.Ordinal) ? "openai/whisper-large-v2" : "openai/whisper-large-v3";
        }
        if (lower.Contains("medium", StringComparison.Ordinal))
        {
            return "openai/whisper-medium";
        }
        if (lower.Contains("small", StringComparison.Ordinal))
        {
            return "openai/whisper-small";
        }
        if (lower.Contains("tiny", StringComparison.Ordinal))
        {
            return "openai/whisper-tiny";
        }
        return "openai/whisper-base";
    }

    /// <summary>distil-whisper variant → HF repo; unlike <see cref="ResolveWhisperRepo"/> this always resolves into
    /// the distil-whisper family because its ids are bare ("large-v3", "large-v3.5").</summary>
    private static string ResolveDistilWhisperRepo(string variant)
    {
        string id = (variant ?? string.Empty).Trim();
        if (id.Contains('/', StringComparison.Ordinal))
        {
            return id;
        }
        string lower = id.ToLowerInvariant();
        if (lower.Contains("v3.5", StringComparison.Ordinal))
        {
            return "distil-whisper/distil-large-v3.5";   // before v3 (v3.5 contains "v3")
        }
        if (lower.Contains("v2", StringComparison.Ordinal))
        {
            return "distil-whisper/distil-large-v2";
        }
        if (lower.Contains("medium", StringComparison.Ordinal))
        {
            return "distil-whisper/distil-medium.en";
        }
        if (lower.Contains("small", StringComparison.Ordinal))
        {
            return "distil-whisper/distil-small.en";
        }
        if (lower.Contains("v3", StringComparison.Ordinal))
        {
            return "distil-whisper/distil-large-v3";
        }
        return "distil-whisper/distil-large-v3.5";
    }

    /// <summary>Moonshine variant → HF repo (only tiny/base exist upstream).</summary>
    private static string ResolveMoonshineRepo(string variant)
    {
        string id = (variant ?? string.Empty).Trim();
        if (id.Contains('/', StringComparison.Ordinal))
        {
            return id;
        }
        return id.Contains("tiny", StringComparison.OrdinalIgnoreCase)
            ? "UsefulSensors/moonshine-tiny"
            : "UsefulSensors/moonshine-base";
    }
}
