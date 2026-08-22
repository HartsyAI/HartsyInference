using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Moonshine;

/// <summary>2nd-gen streaming Moonshine encoder — a completely different front end and attention pattern from the original <see cref="MoonshineEncoder"/> (verified against <c>transformers/models/moonshine_streaming/modeling_moonshine_streaming.py</c> 5.14.1, cross-checked against the real <c>UsefulSensors/moonshine-streaming-{tiny,small,medium}</c> checkpoints).</summary>
// Front end (MoonshineStreamingEncoderEmbedder): reshape raw 16kHz samples into 5ms (80-sample) frames →
// per-frame CMVN (zero-mean, unit-RMS) → learned asinh compression (asinh(exp(log_k) * x)) →
// Linear(80 → hidden, no bias) + SiLU → 2x LEFT-PADDED CAUSAL Conv1d(k=5, s=2) — first with a trailing
// SiLU, the second with none. Total downsample 80x2x2=320 samples/frame ≈ 50 Hz encoder rate (vs the
// original's raw-3-conv 384x stem).
//
// Layers: pre-norm self-attention (per-layer SLIDING-WINDOW mask, no RoPE at all — the original
// encoder's RoPE is absent here) → pre-norm GELU MLP (not gated). Norms are
// MoonshineStreamingOps.UnitOffsetLayerNorm (parameterless LayerNorm × (gamma+1)), NOT the original's
// plain weight-only MoonshineOps.LayerNormNoBias.
//
// This is a full-utterance BATCH forward (bit-exact against the HF reference, which is itself
// batch-only) — true chunked/incremental encoder streaming (exploiting the bounded left/right window per
// layer) is a follow-up; see the class docs on MoonshineStreamingPipeline.
public sealed unsafe class MoonshineStreamingEncoder : IDisposable
{
    private readonly MoonshineConfig _cfg;
    private readonly MoonshineStreamingEncoderLayer[] _layers;

    private Tensor? _logK;
    private Tensor? _conv1Weight, _conv1Bias, _conv2Weight, _conv2Bias;
    private Tensor? _linearWeight;
    private Tensor? _finalNormGamma;
    private bool _loaded;
    private int _disposed;

    public MoonshineConfig Config => _cfg;

    public MoonshineStreamingEncoder(MoonshineConfig cfg)
    {
        if (cfg.SlidingWindows is null || cfg.SlidingWindows.Length != cfg.EncoderLayers)
            throw new ArgumentException("MoonshineConfig.SlidingWindows must be set with one entry per EncoderLayers for the streaming encoder.", nameof(cfg));
        _cfg = cfg;
        _layers = new MoonshineStreamingEncoderLayer[cfg.EncoderLayers];
        for (int i = 0; i < cfg.EncoderLayers; i++) _layers[i] = new MoonshineStreamingEncoderLayer(cfg, cfg.SlidingWindows[i]);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "model.encoder")
    {
        _logK = WhisperOps.EnsureF32(weights[$"{prefix}.embedder.comp.log_k"]);
        _conv1Weight = WhisperOps.EnsureF32(weights[$"{prefix}.embedder.conv1.weight"]);
        _conv1Bias = WhisperOps.EnsureF32(weights[$"{prefix}.embedder.conv1.bias"]);
        _conv2Weight = WhisperOps.EnsureF32(weights[$"{prefix}.embedder.conv2.weight"]);
        _conv2Bias = WhisperOps.EnsureF32(weights[$"{prefix}.embedder.conv2.bias"]);
        _linearWeight = WhisperOps.EnsureF32(weights[$"{prefix}.embedder.linear.weight"]);

        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(weights, $"{prefix}.layers.{i}");

        _finalNormGamma = WhisperOps.EnsureF32(weights[$"{prefix}.final_norm.gamma"]);
        _loaded = true;
    }

    /// <summary><paramref name="audio"/> is a mono 16-kHz waveform in [-1, 1]; length must be a multiple of the frame length (80 samples at the default 5ms/16kHz). Returns <c>[1, T, EncoderHiddenSize]</c>.</summary>
    public Tensor Forward(IBackend backend, float[] audio)
    {
        ThrowIfDisposed();
        if (!_loaded) throw new InvalidOperationException("Call LoadWeights before Forward.");

        int frameLen = (int)Math.Round(_cfg.SampleRate * _cfg.FrameMs / 1000.0);
        int t0 = audio.Length / frameLen;
        if (t0 == 0) throw new ArgumentException($"Audio too short: need at least {frameLen} samples.");
        int enc = _cfg.EncoderHiddenSize;

        Tensor frames = new(new TensorShape(1, t0, frameLen), DType.F32);
        float* fp = (float*)frames.DataPointer;
        for (int i = 0; i < t0 * frameLen; i++) fp[i] = audio[i];

        Tensor cmvn = new(frames.Shape, DType.F32);
        MoonshineStreamingOps.FrameCmvn(cmvn, frames, frameLen);
        frames.Dispose();

        float logK = *(float*)_logK!.DataPointer;
        MoonshineStreamingOps.AsinhCompressInPlace(cmvn, logK);

        Tensor projected = new(new TensorShape(1, t0, enc), DType.F32);
        backend.Linear(projected, cmvn, _linearWeight!, null);
        cmvn.Dispose();
        MoonshineStreamingOps.SiluInPlace(projected);

        // → channels-first for the causal conv stem.
        Tensor chFirst = new(new TensorShape(1, enc, t0), DType.F32);
        backend.Transpose2D(chFirst, projected, t0, enc);
        projected.Dispose();

        int kernel = _cfg.StreamingConvKernel, stride = _cfg.StreamingConvStride, leftPad = kernel - 1;
        int t1 = (t0 + leftPad - kernel) / stride + 1;
        Tensor c1 = new(new TensorShape(1, 2 * enc, t1), DType.F32);
        backend.Conv1d(c1, chFirst, _conv1Weight!, _conv1Bias, stride, leftPad, 0, 1, 1);
        chFirst.Dispose();
        MoonshineStreamingOps.SiluInPlace(c1);

        int t2 = (t1 + leftPad - kernel) / stride + 1;
        Tensor c2 = new(new TensorShape(1, enc, t2), DType.F32);
        backend.Conv1d(c2, c1, _conv2Weight!, _conv2Bias, stride, leftPad, 0, 1, 1);
        c1.Dispose();
        // No activation after conv2 (matches the reference exactly).

        Tensor hidden = new(new TensorShape(1, t2, enc), DType.F32);
        backend.Transpose2D(hidden, c2, enc, t2);
        c2.Dispose();

        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, hidden, t2);
            hidden.Dispose();
            hidden = next;
        }

        Tensor normed = new(hidden.Shape, DType.F32);
        MoonshineStreamingOps.UnitOffsetLayerNorm(normed, hidden, _finalNormGamma!);
        hidden.Dispose();
        return normed;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _logK, _conv1Weight, _conv1Bias, _conv2Weight, _conv2Bias, _linearWeight })
            if (t is not null) yield return t;
        foreach (MoonshineStreamingEncoderLayer layer in _layers)
            foreach (Tensor t in layer.EnumerateWeights()) yield return t;
        if (_finalNormGamma is not null) yield return _finalNormGamma;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(MoonshineStreamingEncoder));
    }

    public void Dispose() { Interlocked.Exchange(ref _disposed, 1); }
}

/// <summary>One streaming encoder block: unit-offset-LN → sliding-window self-attention (no RoPE) → residual → unit-offset-LN → plain (non-gated) GELU MLP → residual.</summary>
internal sealed unsafe class MoonshineStreamingEncoderLayer
{
    private readonly MoonshineConfig _cfg;
    private readonly (int Left, int Right) _window;

    private Tensor? _inputLnGamma;
    private Tensor? _qW, _kW, _vW, _oW;
    private Tensor? _postAttnLnGamma;
    private Tensor? _fc1W, _fc1B, _fc2W, _fc2B;

    public MoonshineStreamingEncoderLayer(MoonshineConfig cfg, (int Left, int Right) window)
    {
        _cfg = cfg;
        _window = window;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _inputLnGamma = WhisperOps.EnsureF32(weights[$"{prefix}.input_layernorm.gamma"]);
        _qW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.q_proj.weight"]);
        _kW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.k_proj.weight"]);
        _vW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.v_proj.weight"]);
        _oW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.o_proj.weight"]);
        _postAttnLnGamma = WhisperOps.EnsureF32(weights[$"{prefix}.post_attention_layernorm.gamma"]);
        _fc1W = WhisperOps.EnsureF32(weights[$"{prefix}.mlp.fc1.weight"]);
        _fc1B = WhisperOps.EnsureF32(weights[$"{prefix}.mlp.fc1.bias"]);
        _fc2W = WhisperOps.EnsureF32(weights[$"{prefix}.mlp.fc2.weight"]);
        _fc2B = WhisperOps.EnsureF32(weights[$"{prefix}.mlp.fc2.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor hidden, int seqLen)
    {
        int d = _cfg.EncoderHiddenSize, heads = _cfg.EncoderNumHeads, hd = _cfg.EncoderHeadDim;
        int qkvDim = heads * hd;
        TensorShape shape = new(1, seqLen, d);

        Tensor normed = new(shape, DType.F32);
        MoonshineStreamingOps.UnitOffsetLayerNorm(normed, hidden, _inputLnGamma!);

        Tensor q = WhisperOps.ProjectLinear(backend, normed, _qW!, null, 1, seqLen, d, qkvDim);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _kW!, null, 1, seqLen, d, qkvDim);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _vW!, null, 1, seqLen, d, qkvDim);
        normed.Dispose();

        TensorShape mh = new(1, heads, seqLen, hd);
        Tensor qMh = new(mh, DType.F32);
        Tensor kMh = new(mh, DType.F32);
        Tensor vMh = new(mh, DType.F32);
        WhisperOps.ReshapeToMultiHead4D(qMh, q, 1, seqLen, heads, hd);
        WhisperOps.ReshapeToMultiHead4D(kMh, k, 1, seqLen, heads, hd);
        WhisperOps.ReshapeToMultiHead4D(vMh, v, 1, seqLen, heads, hd);
        q.Dispose(); k.Dispose(); v.Dispose();
        // No RoPE — the streaming encoder has no positional embedding of any kind at this stage.

        Tensor mask = MoonshineStreamingOps.BuildSlidingWindowMask(seqLen, _window.Left, _window.Right);
        float scale = 1f / MathF.Sqrt(hd);
        Tensor attnOut = new(mh, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, mask, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose(); mask.Dispose();

        Tensor merged = new(shape, DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(merged, attnOut, 1, seqLen, heads, hd);
        attnOut.Dispose();
        Tensor projected = WhisperOps.ProjectLinear(backend, merged, _oW!, null, 1, seqLen, qkvDim, d);
        merged.Dispose();
        Tensor res1 = new(shape, DType.F32);
        backend.Add(res1, hidden, projected);
        projected.Dispose();

        Tensor normed2 = new(shape, DType.F32);
        MoonshineStreamingOps.UnitOffsetLayerNorm(normed2, res1, _postAttnLnGamma!);
        int inter = _cfg.EncoderIntermediateSize;
        Tensor fc1 = WhisperOps.ProjectLinear(backend, normed2, _fc1W!, _fc1B, 1, seqLen, d, inter);
        normed2.Dispose();
        Tensor activated = new(new TensorShape(1, seqLen, inter), DType.F32);
        backend.Gelu(activated, fc1);
        fc1.Dispose();
        Tensor fc2 = WhisperOps.ProjectLinear(backend, activated, _fc2W!, _fc2B, 1, seqLen, inter, d);
        activated.Dispose();
        Tensor res2 = new(shape, DType.F32);
        backend.Add(res2, res1, fc2);
        res1.Dispose(); fc2.Dispose();
        return res2;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_inputLnGamma, _qW, _kW, _vW, _oW, _postAttnLnGamma, _fc1W, _fc1B, _fc2W, _fc2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
