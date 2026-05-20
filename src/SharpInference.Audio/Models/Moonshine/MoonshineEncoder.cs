using SharpInference.Audio.Models.Whisper;  // Reuse WhisperOps (Linear, MultiHead reshapes)
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Moonshine;

/// <summary>Moonshine audio encoder. Operates directly on the raw 16-kHz waveform —
/// no mel spectrogram. The 3-layer Conv1D stem does an aggressive 384× temporal
/// downsample (kernel 127 stride 64 → kernel 7 stride 3 → kernel 3 stride 2), then
/// 8 RoPE transformer blocks produce per-frame hidden states that the decoder
/// cross-attends to.
///
/// <para><b>Layout choices:</b>
/// <list type="bullet">
///   <item>Conv1D is dispatched via the Conv2D H=1 trick (matches our Whisper encoder).</item>
///   <item>Tanh between conv stages is inlined since <c>IBackend</c> doesn't expose it
///         and the audio-stem tensors are small (a few thousand floats).</item>
///   <item>GroupNorm with num_groups=1 (= LayerNorm over [C×T] per sample) is dispatched
///         through <see cref="IBackend.GroupNorm"/> with the spatial axis flattened to
///         [1, C, 1, T].</item>
///   <item>All transformer norms are RMSNorm (weight-only — Llama-style); the bias-less
///         q/k/v/o projections route through <see cref="WhisperOps.ProjectLinear"/> with
///         <c>bias: null</c>.</item>
/// </list></para></summary>
public sealed unsafe class MoonshineEncoder : IDisposable
{
    private readonly MoonshineConfig _cfg;
    private readonly MoonshineEncoderLayer[] _layers;

    private Tensor? _conv1Weight, _conv2Weight, _conv2Bias, _conv3Weight, _conv3Bias;
    private Tensor? _groupNormWeight, _groupNormBias;
    private Tensor? _finalLnWeight;

    private float[]? _cosTable, _sinTable;
    private bool _loaded;
    private int _disposed;

    /// <summary>Config the encoder was built with.</summary>
    public MoonshineConfig Config => _cfg;

    public MoonshineEncoder(MoonshineConfig cfg)
    {
        _cfg = cfg;
        _layers = new MoonshineEncoderLayer[cfg.EncoderLayers];
        for (int i = 0; i < cfg.EncoderLayers; i++) _layers[i] = new MoonshineEncoderLayer(cfg);
    }

    /// <summary>Loads weights from a HuggingFace safetensors dictionary using the
    /// standard <c>model.encoder.*</c> prefix.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "model.encoder")
    {
        Tensor c1 = WhisperOps.EnsureF32(weights[$"{prefix}.conv1.weight"]);          // [out, in, k]
        Tensor c2 = WhisperOps.EnsureF32(weights[$"{prefix}.conv2.weight"]);
        Tensor c3 = WhisperOps.EnsureF32(weights[$"{prefix}.conv3.weight"]);
        _conv1Weight = c1.Reshape(new TensorShape(_cfg.Conv1OutChannels, 1, 1, _cfg.Conv1Kernel));
        _conv2Weight = c2.Reshape(new TensorShape(_cfg.Conv2OutChannels, _cfg.HiddenSize, 1, _cfg.Conv2Kernel));
        _conv3Weight = c3.Reshape(new TensorShape(_cfg.Conv3OutChannels, _cfg.Conv2OutChannels, 1, _cfg.Conv3Kernel));
        _conv2Bias = WhisperOps.EnsureF32(weights[$"{prefix}.conv2.bias"]);
        _conv3Bias = WhisperOps.EnsureF32(weights[$"{prefix}.conv3.bias"]);
        // conv1 has no bias.

        _groupNormWeight = WhisperOps.EnsureF32(weights[$"{prefix}.groupnorm.weight"]);
        _groupNormBias = WhisperOps.EnsureF32(weights[$"{prefix}.groupnorm.bias"]);

        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(weights, $"{prefix}.layers.{i}");

        _finalLnWeight = WhisperOps.EnsureF32(weights[$"{prefix}.layer_norm.weight"]);

        // Precompute RoPE tables. The encoder doesn't have a fixed sequence-length cap,
        // so we size the table generously to 4096 — 4096 / 41.67Hz ≈ 98 s of audio,
        // far longer than typical Moonshine input.
        (float[] cos, float[] sin) = RotaryEmbedding.GetTables(_cfg.RotaryDim, _cfg.RopeTheta, maxPos: 4096);
        _cosTable = cos;
        _sinTable = sin;
        _loaded = true;
    }

    /// <summary>Forward pass. <paramref name="audio"/> is a mono 16-kHz waveform in
    /// [-1, 1] of arbitrary length; the encoder produces
    /// <c>[1, ceil(len / 384), hidden_size]</c> hidden states.</summary>
    public Tensor Forward(IBackend backend, float[] audio)
    {
        ThrowIfDisposed();
        if (!_loaded) throw new InvalidOperationException("Call LoadWeights before Forward.");

        int n = audio.Length;
        // Treat audio as [1, 1, 1, N] so the conv stem flows through IBackend.Conv2D.
        Tensor x = new(new TensorShape(1, 1, 1, n), DType.F32);
        float* xp = (float*)x.DataPointer;
        for (int i = 0; i < n; i++) xp[i] = audio[i];

        // Conv1: [1, 1, 1, N] → [1, 416, 1, T1]. No bias, no padding.
        int t1 = (n - _cfg.Conv1Kernel) / _cfg.Conv1Stride + 1;
        Tensor c1Out = new(new TensorShape(1, _cfg.Conv1OutChannels, 1, t1), DType.F32);
        backend.Conv2D(c1Out, x, _conv1Weight!, bias: null, strideH: 1, strideW: _cfg.Conv1Stride, padH: 0, padW: 0);
        x.Dispose();

        // Tanh after conv1 (matches HF reference).
        MoonshineOps.TanhInPlace(c1Out);

        // GroupNorm(num_groups=1, num_channels=hidden) on [1, hidden, 1, T1].
        Tensor gn = new(c1Out.Shape, DType.F32);
        backend.GroupNorm(gn, c1Out, _groupNormWeight!, _groupNormBias!, groups: 1, eps: 1e-5f);
        c1Out.Dispose();

        // Conv2: [1, hidden, 1, T1] → [1, 2*hidden, 1, T2].
        int t2 = (t1 - _cfg.Conv2Kernel) / _cfg.Conv2Stride + 1;
        Tensor c2Out = new(new TensorShape(1, _cfg.Conv2OutChannels, 1, t2), DType.F32);
        backend.Conv2D(c2Out, gn, _conv2Weight!, _conv2Bias, strideH: 1, strideW: _cfg.Conv2Stride, padH: 0, padW: 0);
        gn.Dispose();
        // GELU after conv2 (NOT tanh — the HF reference uses tanh only after conv1).
        MoonshineOps.GeluInPlace(c2Out);

        // Conv3: [1, 2*hidden, 1, T2] → [1, hidden, 1, T3].
        int t3 = (t2 - _cfg.Conv3Kernel) / _cfg.Conv3Stride + 1;
        Tensor c3Out = new(new TensorShape(1, _cfg.Conv3OutChannels, 1, t3), DType.F32);
        backend.Conv2D(c3Out, c2Out, _conv3Weight!, _conv3Bias, strideH: 1, strideW: _cfg.Conv3Stride, padH: 0, padW: 0);
        c2Out.Dispose();
        MoonshineOps.GeluInPlace(c3Out);

        // Drop H=1 axis and transpose [B, C, T] → [B, T, C].
        Tensor squeezed = c3Out.Reshape(new TensorShape(1, _cfg.HiddenSize, t3));
        Tensor hidden = new(new TensorShape(1, t3, _cfg.HiddenSize), DType.F32);
        backend.Transpose2D(hidden, squeezed, _cfg.HiddenSize, t3);
        c3Out.Dispose();
        // squeezed shares memory with c3Out (Reshape view); freed.

        // 8 RoPE transformer blocks.
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, hidden, t3, _cosTable!, _sinTable!);
            hidden.Dispose();
            hidden = next;
        }

        // Final LayerNorm (bias=False).
        Tensor normed = new(hidden.Shape, DType.F32);
        MoonshineOps.LayerNormNoBias(normed, hidden, _finalLnWeight!, _cfg.LayerNormEps);
        hidden.Dispose();
        return normed;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_conv1Weight is not null) yield return _conv1Weight;
        if (_conv2Weight is not null) yield return _conv2Weight;
        if (_conv2Bias is not null) yield return _conv2Bias;
        if (_conv3Weight is not null) yield return _conv3Weight;
        if (_conv3Bias is not null) yield return _conv3Bias;
        if (_groupNormWeight is not null) yield return _groupNormWeight;
        if (_groupNormBias is not null) yield return _groupNormBias;
        foreach (MoonshineEncoderLayer layer in _layers)
            foreach (Tensor t in layer.EnumerateWeights()) yield return t;
        if (_finalLnWeight is not null) yield return _finalLnWeight;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(MoonshineEncoder));
    }

    public void Dispose() { Interlocked.Exchange(ref _disposed, 1); }
}

/// <summary>Single Moonshine encoder block. Pre-norm self-attention with partial RoPE,
/// then pre-norm GELU MLP. All linear projections are bias-less (attention_bias=false);
/// the MLP fcs DO have biases.</summary>
internal sealed unsafe class MoonshineEncoderLayer
{
    private readonly MoonshineConfig _cfg;

    private Tensor? _inputLnW;
    private Tensor? _qW, _kW, _vW, _oW;
    private Tensor? _postAttnLnW;
    private Tensor? _fc1W, _fc1B, _fc2W, _fc2B;

    public MoonshineEncoderLayer(MoonshineConfig cfg) { _cfg = cfg; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _inputLnW = WhisperOps.EnsureF32(weights[$"{prefix}.input_layernorm.weight"]);
        _qW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.q_proj.weight"]);
        _kW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.k_proj.weight"]);
        _vW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.v_proj.weight"]);
        _oW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.o_proj.weight"]);
        _postAttnLnW = WhisperOps.EnsureF32(weights[$"{prefix}.post_attention_layernorm.weight"]);
        _fc1W = WhisperOps.EnsureF32(weights[$"{prefix}.mlp.fc1.weight"]);
        _fc1B = WhisperOps.EnsureF32(weights[$"{prefix}.mlp.fc1.bias"]);
        _fc2W = WhisperOps.EnsureF32(weights[$"{prefix}.mlp.fc2.weight"]);
        _fc2B = WhisperOps.EnsureF32(weights[$"{prefix}.mlp.fc2.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor hidden, int seqLen, float[] cos, float[] sin)
    {
        int d = _cfg.HiddenSize;
        TensorShape shape = new(1, seqLen, d);

        // Self-attention.
        Tensor normed = new(shape, DType.F32);
        MoonshineOps.LayerNormNoBias(normed, hidden, _inputLnW!, _cfg.LayerNormEps);

        Tensor q = WhisperOps.ProjectLinear(backend, normed, _qW!, bias: null, 1, seqLen, d, d);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _kW!, bias: null, 1, seqLen, d, d);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _vW!, bias: null, 1, seqLen, d, d);
        normed.Dispose();

        TensorShape mh = new(1, _cfg.NumHeads, seqLen, _cfg.HeadDim);
        Tensor qMh = new(mh, DType.F32);
        Tensor kMh = new(mh, DType.F32);
        Tensor vMh = new(mh, DType.F32);
        WhisperOps.ReshapeToMultiHead4D(qMh, q, 1, seqLen, _cfg.NumHeads, _cfg.HeadDim);
        WhisperOps.ReshapeToMultiHead4D(kMh, k, 1, seqLen, _cfg.NumHeads, _cfg.HeadDim);
        WhisperOps.ReshapeToMultiHead4D(vMh, v, 1, seqLen, _cfg.NumHeads, _cfg.HeadDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        // Apply RoPE in place to Q and K (partial — only the first rotary_dim of each head).
        RotaryEmbedding.ApplyInPlace(qMh, _cfg.NumHeads, seqLen, _cfg.HeadDim, _cfg.RotaryDim, posStart: 0, cos, sin);
        RotaryEmbedding.ApplyInPlace(kMh, _cfg.NumHeads, seqLen, _cfg.HeadDim, _cfg.RotaryDim, posStart: 0, cos, sin);

        float scale = 1f / MathF.Sqrt(_cfg.HeadDim);
        Tensor attnOut = new(mh, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, mask: null, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor merged = new(shape, DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(merged, attnOut, 1, seqLen, _cfg.NumHeads, _cfg.HeadDim);
        attnOut.Dispose();

        Tensor projected = WhisperOps.ProjectLinear(backend, merged, _oW!, bias: null, 1, seqLen, d, d);
        merged.Dispose();
        Tensor res1 = new(shape, DType.F32);
        backend.Add(res1, hidden, projected);
        projected.Dispose();

        // MLP (standard GELU).
        Tensor normed2 = new(shape, DType.F32);
        MoonshineOps.LayerNormNoBias(normed2, res1, _postAttnLnW!, _cfg.LayerNormEps);
        Tensor fc1 = WhisperOps.ProjectLinear(backend, normed2, _fc1W!, _fc1B, 1, seqLen, d, _cfg.IntermediateSize);
        normed2.Dispose();
        Tensor activated = new(new TensorShape(1, seqLen, _cfg.IntermediateSize), DType.F32);
        backend.Gelu(activated, fc1);
        fc1.Dispose();
        Tensor fc2 = WhisperOps.ProjectLinear(backend, activated, _fc2W!, _fc2B, 1, seqLen, _cfg.IntermediateSize, d);
        activated.Dispose();
        Tensor res2 = new(shape, DType.F32);
        backend.Add(res2, res1, fc2);
        res1.Dispose(); fc2.Dispose();
        return res2;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_inputLnW, _qW, _kW, _vW, _oW, _postAttnLnW, _fc1W, _fc1B, _fc2W, _fc2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
