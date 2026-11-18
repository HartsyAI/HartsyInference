using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Haru;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.Sampling;

namespace HartsyInference.Engine.Audio;

/// <summary>Haru-4B (<c>higurashi-ai/Haru-4B</c>): a full-duplex speech-to-speech model. One Qwen3.5-4B trunk reads the user's audio and its own
/// text and audio every 133 ms and emits the next text token and the next audio latent; the VibeVoice acoustic codec turns audio into latents
/// and back. A pretrained base model, released for research: it holds a natural conversation but does not follow instructions. Apache-2.0
/// for the weights and the code; the codec is the MIT-licensed VibeVoice acoustic tokenizer, whose card forbids voice impersonation without
/// recorded consent and real-time voice conversion.
///
/// <para>The variant picks the precision of the language model's matrices: native (BF16, 10.5 GB for the whole checkpoint), or <c>q8</c>,
/// <c>q6</c>, <c>q5</c> or <c>q4</c>, converted once into a disk cache. <c>q8</c> fits a 12 GB card (8.1 GB peak measured on an RTX 3060).</para></summary>
internal static class HaruModel
{
    internal const string Repo = "higurashi-ai/Haru-4B";

    /// <summary>The catalog ids this model registers under; a bare id reaches the descriptor as its own variant, which means native.</summary>
    internal static readonly string[] CatalogIds = ["haru", "haru-4b"];

    private const string Category = "tts";
    private const string ConfigFile = "config.json";
    private const string TokenizerFile = "tokenizer.json";
    private const string WeightsFile = "model.safetensors";

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => Repo,
        ResolveCacheTag = variant => ParsePrecision(variant) is HaruPrecision.Native ? "" : ParsePrecision(variant).ToString().ToLowerInvariant(),
        ResolveFiles = (_, _) => Task.FromResult(Files()),
        IsDuplex = true,
        LoadAsync = LoadAsync,
    };

    /// <summary>The files of the pinned release (<c>8d6e83bc</c>): the config, the tokenizer and the weights last.</summary>
    internal static IReadOnlyList<AudioModelFile> Files() =>
    [
        new(ConfigFile, Sha256: "ff29b2dab25780d33fe1ea8efc3d1364d31e7bef95b0b08514b935305ba2f427"),
        new(TokenizerFile, Sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
        new(WeightsFile, Sha256: "ad2c3534ef6bd88525e22e5da3ac4fb739368406bf3b1c852cbafbade647c0fa"),
    ];

    internal static HaruPrecision ParsePrecision(string variant)
    {
        string name = CatalogIds.Contains(variant, StringComparer.OrdinalIgnoreCase) ? "" : variant;
        return HaruWeights.TryParsePrecision(name, out HaruPrecision precision)
            ? precision
            : throw new ArgumentException($"Unknown Haru precision '{variant}'; expected native, q8, q6, q5 or q4 (for example 'haru:q8').", nameof(variant));
    }

    private static async Task<ITtsRunner> LoadAsync(TtsLoadContext context, string variant, CancellationToken cancel)
    {
        HaruPrecision precision = ParsePrecision(variant);
        IReadOnlyDictionary<string, string> paths = await AudioModelCache.FetchAllAsync(Repo, Files(), Category, ct: cancel).ConfigureAwait(false);
        string cache = Path.Combine(AudioModelCache.CacheRoot, Category, "_gguf-cache",
            $"{Repo.Replace('/', '_')}-{precision.ToString().ToLowerInvariant()}-v{HaruWeights.CacheVersion}.gguf");
        HaruPipeline pipeline = await Task.Run(() => HaruPipeline.Load(context.Backend, paths[ConfigFile], paths[TokenizerFile], paths[WeightsFile],
            precision, cache), cancel).ConfigureAwait(false);
        Logs.Info($"[Audio][Haru] Ready at {precision}.");
        return new HaruRunner(pipeline);
    }

    /// <summary>The loaded model as a speech runner that only opens conversations.</summary>
    private sealed class HaruRunner(HaruPipeline pipeline) : IDuplexSpeechRunner
    {
        public int SampleRate => pipeline.SampleRate;

        public float[] Synthesize(IBackend backend, TtsJob job) =>
            throw new NotSupportedException("Haru-4B is a full-duplex speech-to-speech model: it answers a user's audio and does not read text aloud. "
                + "Open a conversation with ISpeechService.OpenDuplexAsync, or run 'hartsy duplex'.");

        public IDuplexSpeechSession Open(DuplexSpeechRequest request, float[]? referenceMono24k)
        {
            HaruDuplexOptions options = new()
            {
                GuidanceScale = request.Guidance, Temperature = request.Temperature, TopK = request.TopK, TopP = request.TopP,
                Dry = request.RepetitionPenalty ? new DryOptions() : null,
                EopBias = request.EopBias, SpeakFirst = request.SpeakFirst, MaxFrames = request.MaxFrames, Seed = request.Seed,
            };
            float[] reference = referenceMono24k is { Length: > 0 } clip ? HaruPipeline.PrepareReference(clip) : [];
            return new Session(pipeline.OpenSession(options), request.Card, reference, pipeline.SampleRate, pipeline.FrameSamples);
        }

        public void Dispose() => pipeline.Dispose();
    }

    private sealed class Session(HaruDuplexSession inner, string card, float[] reference, int sampleRate, int frameSamples) : IDuplexSpeechSession
    {
        public int SampleRate => sampleRate;

        public int FrameSamples => frameSamples;

        public int RemainingFrames => inner.RemainingFrames;

        public DuplexFrame Start() => Map(inner.Start(card, reference));

        public DuplexFrame Step(ReadOnlySpan<float> userFrame) => Map(inner.Step(userFrame));

        public void Dispose() => inner.Dispose();

        private static DuplexFrame Map(HaruFrame frame) => new(frame.Audio, frame.Text, frame.Index, frame.Seconds, frame.Done);
    }
}
