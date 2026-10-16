using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Fish Audio Drama 3 text-to-speech, ported from fish-speech's <c>generate_long_drama3</c>. The text is cut into chunks
/// (<see cref="FishAudioDrama3Chunking"/>), each generated on its own with the voice references, the caller's earlier turns and as
/// much of the conversation so far as fits (the first chunk pinned, plus the latest), and decoded by S2's ModifiedDAC codec to
/// 44.1 kHz mono. Every chunk is reported as a finished turn so a client can send a take back as history to revise it.</summary>
public sealed class FishAudioDrama3Pipeline : IDisposable
{
    /// <summary>Frames before the first streamed audio; later emissions double the decoded span, up to <see cref="StreamMaxStep"/>.</summary>
    public const int StreamFirstFrames = 16;

    /// <summary>Most frames between two streamed emissions.</summary>
    public const int StreamMaxStep = 256;

    private readonly FishAudioDrama3Config _config;
    private readonly ModifiedDacDecoder _codec;
    private readonly ModifiedDacEncoder _encoder;
    private readonly FishAudioDrama3Prompt _prompt;
    private FishAudioDrama3Lm? _lm;
    private int _disposed;

    /// <param name="encode">Tokenizes one literal, parsing special tokens inline and adding none of its own.</param>
    public FishAudioDrama3Pipeline(FishAudioDrama3Config config, ModifiedDacConfig codec, Func<string, int[]> encode)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        ArgumentNullException.ThrowIfNull(codec);
        if (codec.TotalCodebooks != config.NumCodebooks)
        {
            throw new ArgumentException($"The codec has {codec.TotalCodebooks} codebooks; Drama 3 predicts {config.NumCodebooks}.", nameof(codec));
        }
        _codec = new ModifiedDacDecoder(codec);
        _encoder = new ModifiedDacEncoder(codec);
        _prompt = new FishAudioDrama3Prompt(encode, config);
    }

    public int SampleRate => _config.SampleRate;

    public FishAudioDrama3Config Config => _config;

    /// <summary>The loaded model; null before <see cref="LoadWeights"/>.</summary>
    public FishAudioDrama3Lm? Model => _lm;

    /// <summary>Loads the model (checkpoint names, which must outlive the pipeline) and the codec.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> model, IReadOnlyDictionary<string, Tensor> codec, int capacity)
    {
        ThrowIfDisposed();
        _codec.LoadWeights(codec);
        _encoder.LoadWeights(codec);
        _lm?.Dispose();
        _lm = new FishAudioDrama3Lm(_config, model, capacity);
    }

    /// <summary>Turns a reference clip (mono, <see cref="SampleRate"/>) into the codes a reference or history turn carries.</summary>
    public int[,] EncodeReference(IBackend backend, ReadOnlySpan<float> audio) => _encoder.Encode(backend, audio);

    /// <summary>Decodes codes <c>[numCodebooks, frames]</c> to PCM.</summary>
    public float[] Decode(IBackend backend, int[,] codes) => _codec.Decode(backend, codes, codes.GetLength(1));

    /// <summary>One synthesis request. Null sampling fields take the model defaults.</summary>
    public sealed record Request
    {
        public required string Text { get; init; }

        /// <summary>Voice references: each a transcript and its clip's codes, carried as <c>[Reference N]</c> blocks.</summary>
        public IReadOnlyList<FishAudioDrama3Turn> References { get; init; } = [];

        /// <summary>Earlier turns of the conversation (user text and the assistant's codes), always carried before the new text.</summary>
        public IReadOnlyList<FishAudioDrama3Turn> History { get; init; } = [];

        public float? Temperature { get; init; }
        public float? TopP { get; init; }
        public int? TopK { get; init; }

        /// <summary>Frame cap per chunk; null or zero uses <see cref="FishAudioDrama3Chunking.ChunkMaxNewFrames"/>
        /// (<see cref="FishAudioDrama3Config.DefaultMaxNewFrames"/> with <see cref="SingleChunk"/>).</summary>
        public int? MaxFrames { get; init; }

        /// <summary>Generates the whole text in one pass instead of chunking it.</summary>
        public bool SingleChunk { get; init; }

        public int Seed { get; init; }

        public CancellationToken Cancel { get; init; }
    }

    /// <summary>What a request produced: the PCM and one finished turn per generated chunk.</summary>
    /// <param name="Audio">44.1 kHz mono PCM of every chunk, in order.</param>
    /// <param name="Turns">Each chunk's text and codes, for sending back as history.</param>
    public sealed record Result(float[] Audio, IReadOnlyList<FishAudioDrama3Turn> Turns);

    /// <summary>Generates and decodes the request. With <paramref name="onAudio"/>, new audio is handed over while a chunk is still
    /// generating: the codec is causal, so a decode of the frames so far ends exactly where the full decode would have.</summary>
    public Result Synthesize(IBackend backend, Request request, Action<float[]>? onAudio = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(request);
        FishAudioDrama3Lm lm = _lm ?? throw new InvalidOperationException("LoadWeights must run before Synthesize.");
        List<string> chunks = request.SingleChunk ? [request.Text] : FishAudioDrama3Chunking.Split(request.Text);
        if (chunks.Count == 0) chunks = [request.Text];
        // Reject bad text in any chunk before generating: failing at a later chunk would waste the earlier ones.
        foreach (string chunk in chunks)
        {
            int rows = _prompt.Build(chunk, request.References, request.History).GetLength(0);
            if (rows >= lm.Capacity) throw new ArgumentException($"The prompt has {rows} rows; no room is left to generate.");
        }
        int requested = request.MaxFrames is > 0 ? request.MaxFrames.Value
            : request.SingleChunk ? FishAudioDrama3Config.DefaultMaxNewFrames : FishAudioDrama3Chunking.ChunkMaxNewFrames;
        Logs.Info($"[Audio][Drama3] {chunks.Count} chunk(s), {request.References.Count} reference(s), {request.History.Count} history "
            + $"turn(s), up to {requested} frames per chunk.");

        FishAudioDrama3Sampler sampler = new(_config, request.Seed, request.Temperature, request.TopP, request.TopK);
        FishAudioDrama3History chunkHistory = new();
        List<float> audio = [];
        List<FishAudioDrama3Turn> turns = [];
        foreach (string chunk in chunks)
        {
            request.Cancel.ThrowIfCancellationRequested();
            int[,] prompt = _prompt.BuildWithHistory(chunk, request.References, request.History, chunkHistory);
            StreamState? stream = onAudio is null ? null : new StreamState(this, backend, onAudio);
            FishAudioDrama3Generation generation = lm.Generate(backend, prompt, requested, sampler,
                stream is null ? null : stream.OnFrame, request.Cancel);
            if (!generation.ReachedEos)
            {
                Logs.Warning($"[Audio][Drama3] A chunk reached {generation.Frames} frames before eos; its audio may be cut short.");
            }
            if (generation.Frames == 0)
            {
                Logs.Warning("[Audio][Drama3] The model ended a chunk before any audio frame.");
                continue;
            }
            float[] pcm = Decode(backend, generation.Codes);
            stream?.Finish(pcm);
            audio.AddRange(pcm);
            FishAudioDrama3Turn turn = new(chunk, generation.Codes);
            turns.Add(turn);
            chunkHistory.Push(turn);
        }
        return new Result([.. audio], turns);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lm?.Dispose();
        _codec.Dispose();
        _encoder.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed != 0, this);

    /// <summary>Progressive decode of the chunk being generated: each emission decodes every frame so far and hands over the samples
    /// past the last emission; the spans double so the repeated decodes stay a small multiple of one.</summary>
    private sealed class StreamState(FishAudioDrama3Pipeline pipeline, IBackend backend, Action<float[]> onAudio)
    {
        private readonly List<int[]> _frames = [];
        private int _emittedSamples;
        private int _nextEmit = StreamFirstFrames;

        public void OnFrame(ReadOnlySpan<int> codes)
        {
            _frames.Add(codes.ToArray());
            if (_frames.Count < _nextEmit) return;
            int[,] grid = new int[codes.Length, _frames.Count];
            for (int f = 0; f < _frames.Count; f++)
            {
                for (int c = 0; c < codes.Length; c++) grid[c, f] = _frames[f][c];
            }
            Emit(pipeline.Decode(backend, grid));
            _nextEmit = _frames.Count + Math.Min(_frames.Count, StreamMaxStep);
        }

        /// <summary>Hands over what the chunk's full decode adds past the last emission.</summary>
        public void Finish(float[] fullChunk) => Emit(fullChunk);

        private void Emit(float[] decoded)
        {
            if (decoded.Length <= _emittedSamples) return;
            onAudio(decoded[_emittedSamples..]);
            _emittedSamples = decoded.Length;
        }
    }
}
