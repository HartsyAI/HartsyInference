using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Moonshine;

/// <summary>2nd-gen streaming Moonshine decoder — the per-layer block is architecturally IDENTICAL to the original <see cref="MoonshineDecoder"/> (verified against <c>transformers/models/moonshine_streaming/modeling_moonshine_streaming.py</c> 5.14.1), so this wraps and reuses it unchanged, adding only two deltas.</summary>
// 1. A learned absolute positional embedding (model.decoder.pos_emb) is added to the ENCODER hidden
//    states (not the decoder's token embeddings) once, before cross-attention K/V are precomputed — the
//    decoder's own token stream gets no additive positional signal at all (RoPE inside self-attention is
//    its only position source).
// 2. An optional model.decoder.proj Linear (present only when the encoder and decoder hidden sizes
//    differ, e.g. streaming-small: 620→512, streaming-medium: 768→640; absent — Identity — for
//    streaming-tiny where both are 320) re-projects the positioned encoder states to the decoder's
//    hidden size before cross-attention.
// MoonshineConfig.TieWordEmbeddings must be false for streaming configs — the inner MoonshineDecoder
// then loads a separate top-level proj_out.weight LM head.
public sealed unsafe class MoonshineStreamingDecoder : IDisposable
{
    private readonly MoonshineConfig _cfg;
    private readonly MoonshineDecoder _inner;
    private Tensor? _posEmb;
    private Tensor? _projWeight;
    private bool _loaded;
    private int _disposed;

    public MoonshineConfig Config => _cfg;

    public MoonshineStreamingDecoder(MoonshineConfig cfg)
    {
        _cfg = cfg;
        _inner = new MoonshineDecoder(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "model.decoder")
    {
        _inner.LoadWeights(weights, prefix);
        _posEmb = WhisperOps.EnsureF32(weights[$"{prefix}.pos_emb.weight"]);
        _projWeight = weights.TryGetValue($"{prefix}.proj.weight", out Tensor? p) ? WhisperOps.EnsureF32(p) : null;
        _loaded = true;
    }

    /// <summary>Adds the encoder-side positional embedding, optionally re-projects to the decoder's hidden size, then precomputes cross-attention K/V — mirrors <see cref="MoonshineDecoder.StartDecode"/>.</summary>
    public MoonshineDecoder.DecodeState StartDecode(IBackend backend, Tensor encoderHidden)
    {
        ThrowIfDisposed();
        if (!_loaded) throw new InvalidOperationException("Call LoadWeights before StartDecode.");
        int t = (int)encoderHidden.Shape[1];
        int encDim = _cfg.EncoderHiddenSize;

        Tensor withPos = new(encoderHidden.Shape, DType.F32);
        AddPositionalEmbedding(withPos, encoderHidden, _posEmb!, t, encDim);

        Tensor projected;
        bool ownsProjected = _projWeight is not null;
        if (ownsProjected)
        {
            projected = new Tensor(new TensorShape(1, t, _cfg.HiddenSize), DType.F32);
            backend.Linear(projected, withPos, _projWeight!, null);
        }
        else
        {
            projected = withPos;
        }

        MoonshineDecoder.DecodeState state = _inner.StartDecode(backend, projected);
        if (ownsProjected) projected.Dispose();
        withPos.Dispose();
        return state;
    }

    public Tensor DecodeStep(IBackend backend, ReadOnlySpan<int> tokenIds, MoonshineDecoder.DecodeState state)
        => _inner.DecodeStep(backend, tokenIds, state);

    private static void AddPositionalEmbedding(Tensor output, Tensor input, Tensor posEmb, int t, int d)
    {
        float* o = (float*)output.DataPointer;
        float* x = (float*)input.DataPointer;
        float* p = (float*)posEmb.DataPointer;   // rows [0, t) of the [maxPos, d] table
        long n = (long)t * d;
        for (long i = 0; i < n; i++) o[i] = x[i] + p[i];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _inner.EnumerateWeights()) yield return t;
        if (_posEmb is not null) yield return _posEmb;
        if (_projWeight is not null) yield return _projWeight;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(MoonshineStreamingDecoder));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) _inner.Dispose();
    }
}
