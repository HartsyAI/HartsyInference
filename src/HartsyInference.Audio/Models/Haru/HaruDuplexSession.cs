using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Sampling;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>A live full-duplex conversation: one user frame in, one model frame out. <see cref="Start"/> runs the prompt (character card and
/// voice reference) and returns frame 0; each <see cref="Step"/> consumes the user's next 3200 samples and returns the model's next frame.
/// The model stays one frame ahead, so the 133 ms frame is the only structural latency.</summary>
/// <remarks>Frame t of the model comes from the hidden state of the previous position: its text token from the tied embedding, its audio
/// latent from the MeanFlow head, decoded at once. A session borrows the model, head and codec and drives them from one thread.</remarks>
public sealed unsafe class HaruDuplexSession : IDisposable
{
    private const int PrefillChunk = 512;

    private readonly HaruConfig _config;
    private readonly HaruLm _lm;
    private readonly HaruMeanFlowHead _head;
    private readonly HaruCodecSession _codec;
    private readonly HaruTextCodec _text;
    private readonly IBackend _backend;
    private readonly HaruDuplexOptions _options;
    private readonly IHaruNoise _noise;
    private readonly int[] _silentTokens;
    private readonly int[] _bannedTokens;
    private readonly List<int> _history = [];
    private SamplerChain? _sampler;
    private HaruStreamDetokenizer? _detokenizer;
    private float[]? _latent;
    private int _token;
    private int _forceText;
    private readonly Action? _release;
    private int _index;
    private int _disposed;

    /// <summary>Creates a session; <paramref name="noise"/> defaults to a generator seeded with <see cref="HaruDuplexOptions.Seed"/>.</summary>
    internal HaruDuplexSession(HaruConfig config, HaruLm lm, HaruMeanFlowHead head, HaruCodecSession codec, HaruTextCodec text,
        IBackend backend, HaruDuplexOptions options, IHaruNoise? noise = null, Action? release = null)
    {
        _release = release;
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _lm = lm ?? throw new ArgumentNullException(nameof(lm));
        _head = head ?? throw new ArgumentNullException(nameof(head));
        _codec = codec ?? throw new ArgumentNullException(nameof(codec));
        _text = text ?? throw new ArgumentNullException(nameof(text));
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        if (options.GuidanceScale < 1f || options.GuidanceScale > 2f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "The audio guidance scale is valid from 1 to 2.");
        }
        _noise = noise ?? new HaruSeededNoise(options.Seed);
        _silentTokens = [config.PadTokenId, config.EopTokenId];
        _bannedTokens = config.EosTokenId is int eos ? [config.PadTokenId, config.EopTokenId, eos] : _silentTokens;
    }

    /// <summary>Frames produced since <see cref="Start"/>.</summary>
    public int FramesProduced => _index;

    /// <summary>Releases the conversation so the pipeline can open another.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _release?.Invoke();
        }
    }

    /// <summary>Runs the card and the voice reference (prepared mono 24 kHz audio, or empty) and returns the model's frame 0.</summary>
    public HaruFrame Start(string card, ReadOnlySpan<float> reference)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        long started = Stopwatch.GetTimestamp();
        int limit = (int)(_options.ReferenceSeconds * _config.SampleRate);
        if (reference.Length > limit)
        {
            reference = reference[^limit..];
        }
        _lm.Reset();
        _codec.Reset();
        _history.Clear();
        _sampler = BuildSampler();
        _detokenizer = new HaruStreamDetokenizer(_text.Decode, _config.PadTokenId, _config.EopTokenId);
        _forceText = _options.SpeakFirst ? 1 : 0;
        _index = 0;

        HaruPrompt prompt = HaruPrompt.Build(_config, _text, _codec, _noise, card, reference, _options.SpeakFirst);
        if (prompt.Length + 1 > _lm.Capacity)
        {
            throw new ArgumentException($"The prompt takes {prompt.Length} positions of the {_lm.Capacity} this model was loaded for; use a shorter voice reference or card.");
        }
        int d = _config.LatentDim;
        Tensor? last = null;
        for (int start = 0; start < prompt.Length; start += PrefillChunk)
        {
            int rows = Math.Min(PrefillChunk, prompt.Length - start);
            using Tensor embeds = _lm.Embed(_backend, prompt.Tokens.AsSpan(start, rows), prompt.ModelLatents.AsSpan(start * d, rows * d),
                prompt.ModelMask.AsSpan(start, rows), prompt.UserLatents.AsSpan(start * d, rows * d), prompt.UserMask.AsSpan(start, rows));
            using Tensor hidden = _lm.Forward(_backend, embeds, rows);
            last?.Dispose();
            last = LastRow(hidden, rows);
        }
        using Tensor row = last!;
        return Emit(row, started);
    }

    /// <summary>Consumes the user's next frame (3200 samples at 24 kHz) and returns the model's next frame.</summary>
    public HaruFrame Step(ReadOnlySpan<float> userFrame)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_latent is null || _sampler is null)
        {
            throw new InvalidOperationException("Call Start before Step.");
        }
        if (userFrame.Length != _config.FrameSamples)
        {
            throw new ArgumentException($"A user frame is {_config.FrameSamples} samples, got {userFrame.Length}.", nameof(userFrame));
        }
        long started = Stopwatch.GetTimestamp();
        float[] userLatent = _codec.Sample(_codec.EncodeFrame(userFrame), _noise);
        using Tensor embeds = _lm.Embed(_backend, [_token], _latent, [], userLatent, []);
        using Tensor hidden = _lm.Forward(_backend, embeds, 1);
        using Tensor row = LastRow(hidden, 1);
        return Emit(row, started);
    }

    private HaruFrame Emit(Tensor row, long started)
    {
        float[] logits = TextLogits(row);
        int token = _sampler!.Next(logits, _history);

        int d = _config.LatentDim;
        float[] draws = new float[d];
        _noise.FillGaussian(draws);
        float[] latent;
        using (Tensor noise = Make(draws, 1, 1, d))
        using (Tensor normalized = _head.Sample(_backend, row, noise, _options.GuidanceScale))
        {
            latent = new ReadOnlySpan<float>((void*)normalized.DataPointer, d).ToArray();
        }
        for (int i = 0; i < d; i++)
        {
            latent[i] = _lm.Unscale(latent[i]);
        }
        float[] audio = _codec.DecodeFrame(latent);
        for (int i = 0; i < audio.Length; i++)
        {
            audio[i] = Math.Clamp(audio[i], -1f, 1f);
        }

        _token = token;
        _latent = latent;
        _history.Add(token);
        string text = _detokenizer!.Push(token);
        int index = _index++;
        return new HaruFrame(audio, token, text, index, Stopwatch.GetElapsedTime(started).TotalSeconds, _index >= _options.MaxFrames || _lm.Position + 1 >= _lm.Capacity);
    }

    /// <summary>The text logits of a hidden row, with the PAD and EOP logits taken in F32, the EOP bias added and, for the first frame of a
    /// model that speaks first, the silent tokens banned.</summary>
    private float[] TextLogits(Tensor row)
    {
        using Tensor all = _lm.Logits(_backend, row, 1);
        float[] logits = new ReadOnlySpan<float>((void*)all.DataPointer, _lm.VocabSize).ToArray();
        Span<float> exact = stackalloc float[2];
        _lm.StreamLogits(row, 0, _silentTokens, exact);
        logits[_config.PadTokenId] = exact[0];
        logits[_config.EopTokenId] = exact[1];
        if (_options.EopBias != 0f)
        {
            logits[_config.EopTokenId] += _options.EopBias;
        }
        if (_forceText > 0)
        {
            foreach (int banned in _bannedTokens)
            {
                logits[banned] = float.NegativeInfinity;
            }
            _forceText--;
        }
        return logits;
    }

    private SamplerChain BuildSampler()
    {
        List<ISamplerStep> steps = [];
        if (_options.Dry is not null)
        {
            steps.Add(new DryPenaltyStep(_options.Dry, _bannedTokens));
        }
        if (_options.Temperature > 0f && _options.Temperature != 1f)
        {
            steps.Add(new TemperatureStep(_options.Temperature));
        }
        if (_options.TopK > 0)
        {
            steps.Add(new TopKStep(_options.TopK));
        }
        if (_options.TopP < 1f)
        {
            steps.Add(new TopPStep(_options.TopP));
        }
        return SamplerChain.FromSteps(steps, greedy: _options.Temperature <= 0f, seed: (ulong)_options.Seed);
    }

    private Tensor LastRow(Tensor hidden, int rows)
    {
        int h = _lm.HiddenSize;
        Tensor row = new(new TensorShape(1, 1, h), DType.F32);
        new ReadOnlySpan<float>((float*)hidden.DataPointer + (long)(rows - 1) * h, h).CopyTo(new Span<float>((void*)row.DataPointer, h));
        return row;
    }

    private static Tensor Make(ReadOnlySpan<float> values, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        values.CopyTo(new Span<float>((void*)t.DataPointer, values.Length));
        return t;
    }
}
