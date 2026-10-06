using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Fish Audio S2 text-to-speech: builds the conversation prompt, runs the Dual-AR model to the
/// <c>&lt;|im_end|&gt;</c> token, and decodes the generated code grid with the ModifiedDAC decoder. Port of fish-speech's
/// <c>generate_long</c>: text with <c>&lt;|speaker:N|&gt;</c> tags is split into batches, and each batch's generated codes
/// are appended to the conversation as an assistant turn so later batches keep the same voice.</summary>
public sealed class FishAudioS2Pipeline : IDisposable
{
    private readonly FishAudioS2Config _cfg;
    private readonly FishAudioS2DualAr _model;
    private readonly ModifiedDacDecoder _codec;
    private readonly ModifiedDacEncoder _encoder;
    private readonly Func<string, int[]> _encode;
    private int _disposed;

    public FishAudioS2Pipeline(FishAudioS2Config cfg, ModifiedDacConfig codec, Func<string, int[]> encode)
    {
        _cfg = cfg;
        _model = new FishAudioS2DualAr(cfg);
        _codec = new ModifiedDacDecoder(codec);
        _encoder = new ModifiedDacEncoder(codec);
        _encode = encode;
    }

    public int SampleRate => _cfg.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> model, IReadOnlyDictionary<string, Tensor> codec)
    {
        _model.LoadWeights(model);
        _codec.LoadWeights(codec);
        _encoder.LoadWeights(codec);
    }

    /// <summary>Turns a reference clip (mono, <see cref="SampleRate"/>) into the code grid a cloning prompt carries.</summary>
    public int[,] EncodeReference(IBackend backend, ReadOnlySpan<float> audio) => _encoder.Encode(backend, audio);

    /// <summary>One synthesis request. Null sampling fields use the model defaults.</summary>
    public sealed record Request
    {
        public required string Text { get; init; }
        /// <summary>Codes of a reference clip (<c>[numCodebooks, T]</c>) and its transcript, for voice cloning.</summary>
        public int[,]? ReferenceCodes { get; init; }
        public string? ReferenceText { get; init; }
        public float? Temperature { get; init; }
        public float? TopP { get; init; }
        public int? TopK { get; init; }
        /// <summary>Frame cap per batch; 0/null = <see cref="DefaultMaxFrames"/>.</summary>
        public int? MaxFrames { get; init; }
        public int Seed { get; init; }
        public int MaxBatchBytes { get; init; } = 512;
        public CancellationToken Cancel { get; init; }
    }

    public const int DefaultMaxFrames = 2_048;

    /// <summary>Runs generation only, returning one code grid per text batch.</summary>
    public List<int[,]> GenerateCodes(IBackend backend, Request req)
    {
        ThrowIfDisposed();
        FishAudioS2Config cfg = _cfg with
        {
            Temperature = req.Temperature ?? _cfg.Temperature,
            TopP = req.TopP ?? _cfg.TopP,
            TopK = req.TopK ?? _cfg.TopK,
        };
        FishAudioS2Prompt conversation = new(_encode, cfg.SemanticBeginId);
        conversation.System(req.ReferenceText, req.ReferenceCodes);

        List<int[,]> results = [];
        uint rng = DeterministicRng.Seed(req.Seed);
        int max = req.MaxFrames is > 0 ? req.MaxFrames.Value : DefaultMaxFrames;
        foreach (string batch in FishAudioS2Prompt.SplitBatches(req.Text, maxBytes: req.MaxBatchBytes))
        {
            req.Cancel.ThrowIfCancellationRequested();
            conversation.User(batch);
            // The open assistant turn is only part of this generation's prompt, not of the kept conversation.
            FishAudioS2Prompt prompt = Clone(conversation, cfg.SemanticBeginId).OpenAssistant();
            int[,] codes = Generate(backend, cfg, prompt, max, ref rng, req.Cancel);
            if (codes.GetLength(1) == 0) continue;
            conversation.Assistant(codes);
            results.Add(codes);
        }
        return results;
    }

    /// <summary>Generates and decodes; returns 44.1 kHz mono PCM (empty when nothing was generated).</summary>
    public float[] Synthesize(IBackend backend, Request req)
    {
        List<float> audio = [];
        foreach (int[,] codes in GenerateCodes(backend, req))
            audio.AddRange(_codec.Decode(backend, codes, codes.GetLength(1)));
        return [.. audio];
    }

    /// <summary>Decodes a code grid to PCM.</summary>
    public float[] Decode(IBackend backend, int[,] codes) => _codec.Decode(backend, codes, codes.GetLength(1));

    private FishAudioS2Prompt Clone(FishAudioS2Prompt source, int semanticBegin)
    {
        FishAudioS2Prompt copy = new(_encode, semanticBegin);
        copy.CopyFrom(source);
        return copy;
    }

    private int[,] Generate(IBackend backend, FishAudioS2Config cfg, FishAudioS2Prompt prompt, int maxFrames, ref uint rng,
        CancellationToken cancel)
    {
        int n = cfg.NumCodebooks, promptLength = prompt.Length;
        if (promptLength + maxFrames + 2 > cfg.Slow.MaxPositionEmbeddings)
            maxFrames = Math.Max(1, cfg.Slow.MaxPositionEmbeddings - promptLength - 2);
        using IKvCache cache = _model.CreateSlowCache(promptLength + maxFrames + 2);

        int[] tokens = prompt.Tokens;
        using (Tensor embeds = _model.EmbedFrames(tokens, prompt.Codes))
        {
            using Tensor hidden = _model.ForwardHidden(backend, embeds, promptLength, 0, cache);
            return Loop(backend, cfg, hidden, cache, promptLength, maxFrames, n, ref rng, cancel);
        }
    }

    private int[,] Loop(IBackend backend, FishAudioS2Config cfg, Tensor firstHidden, IKvCache cache, int promptLength,
        int maxFrames, int n, ref uint rng, CancellationToken cancel)
    {
        List<int[]> frames = new(maxFrames);
        List<int> recent = [];     // trailing main tokens for Repetition Aware Sampling
        (int token, int[] codes) = _model.SampleFrame(backend, firstHidden, ref rng, null);
        int pos = promptLength;
        while (token != cfg.ImEndId && frames.Count < maxFrames)
        {
            cancel.ThrowIfCancellationRequested();
            frames.Add(codes);
            if (recent.Count == cfg.RasWindow) recent.RemoveAt(0);
            recent.Add(token);

            using Tensor embed = _model.EmbedFrames([token], [codes]);
            using Tensor hidden = _model.ForwardHidden(backend, embed, 1, pos++, cache);
            (token, codes) = _model.SampleFrame(backend, hidden, ref rng, recent);
        }

        int[,] grid = new int[n, frames.Count];
        for (int j = 0; j < frames.Count; j++)
            for (int i = 0; i < n; i++) grid[i, j] = frames[j][i];
        return grid;
    }

    public IEnumerable<Tensor> EnumerateWeights() => _model.EnumerateWeights();

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _model.Dispose(); _codec.Dispose(); _encoder.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(FishAudioS2Pipeline));
    }
}
