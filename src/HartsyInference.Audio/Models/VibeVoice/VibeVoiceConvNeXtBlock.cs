using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VibeVoice;

/// <summary>ConvNeXt-V1-style 1D block used by both the acoustic and semantic VAE
/// encoders, and by the acoustic decoder. Mirrors Python's <c>Block1D</c> in
/// <c>modular_vibevoice_tokenizer.py</c>.
///
/// <para>Per-block forward (channels-first <c>[B, C, T]</c> throughout, transposing only
/// inside the FFN):
/// <code>
///   # mixer
///   r = x
///   x = ConvRMSNorm(x)                # channel-axis RMSNorm
///   x = DepthwiseConv1d(x, k=7)       # causal, groups=C, stride=1
///   x = x * gamma                     # per-channel layer scale
///   x = r + x
///   # ffn
///   r = x
///   x = ConvRMSNorm(x)
///   x = x.transpose(1, 2)             # → [B, T, C]
///   x = Linear(C → 4C)
///   x = GELU(x)
///   x = Linear(4C → C)
///   x = x.transpose(1, 2)             # → [B, C, T]
///   x = x * ffn_gamma
///   x = r + x
/// </code></para>
///
/// <para>FFN expansion is 4× across all official checkpoints. Layer scale γ initializes
/// to 1e-6 — at inference we just load the trained value.</para></summary>
internal sealed unsafe class VibeVoiceConvNeXtBlock
{
    private readonly string _prefix;
    private readonly int _dim;
    private readonly int _ffnDim;
    private readonly float _normEps;

    private readonly SConv1d _mixer;

    private Tensor? _normW;
    private Tensor? _ffnNormW;
    private Tensor? _gammaW;        // layer scale as [dim,1,1] depthwise-conv kernel
    private Tensor? _ffnGammaW;     // FFN layer scale as [dim,1,1] depthwise-conv kernel
    private Tensor? _ffn1W, _ffn1B;
    private Tensor? _ffn2W, _ffn2B;

    public VibeVoiceConvNeXtBlock(string prefix, int dim, int ffnExpansion, int kernelSize, float normEps)
    {
        _prefix = prefix;
        _dim = dim;
        _ffnDim = ffnExpansion * dim;
        _normEps = normEps;
        // Depthwise causal Conv1D: groups=C, stride=1, in=out=dim.
        _mixer = new SConv1d(
            layerId: $"{prefix}.mixer",
            inChannels: dim,
            outChannels: dim,
            kernelSize: kernelSize,
            stride: 1,
            dilation: 1,
            groups: dim,
            bias: true);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _normW = WhisperOps.EnsureF32(w[$"{_prefix}.norm.weight"]);
        _ffnNormW = WhisperOps.EnsureF32(w[$"{_prefix}.ffn_norm.weight"]);
        _gammaW = VibeVoiceOps.ToChannelScaleWeight(w[$"{_prefix}.gamma"], _dim);
        _ffnGammaW = VibeVoiceOps.ToChannelScaleWeight(w[$"{_prefix}.ffn_gamma"], _dim);
        _ffn1W = WhisperOps.EnsureF32(w[$"{_prefix}.ffn.linear1.weight"]);
        _ffn1B = WhisperOps.EnsureF32(w[$"{_prefix}.ffn.linear1.bias"]);
        _ffn2W = WhisperOps.EnsureF32(w[$"{_prefix}.ffn.linear2.weight"]);
        _ffn2B = WhisperOps.EnsureF32(w[$"{_prefix}.ffn.linear2.bias"]);
        _mixer.LoadWeights(w, $"{_prefix}.mixer.conv");
    }

    /// <summary>Forward — <paramref name="x"/> is <c>[batch, dim, t]</c> channels-first.
    /// Returns a freshly-allocated <c>[batch, dim, t]</c> tensor. Caller owns disposal of
    /// the returned tensor; the input is NOT disposed.
    ///
    /// <para>When <paramref name="cache"/> is non-null the mixer runs in streaming mode
    /// (depthwise conv keeps a per-(layer, sample) history). Non-streaming when null.</para></summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int t,
        VibeVoiceTokenizerStreamingCache? cache = null, ReadOnlySpan<int> sampleIndices = default)
    {
        if (_normW is null) throw new InvalidOperationException($"VibeVoiceConvNeXtBlock '{_prefix}' weights not loaded.");

        // --- mixer path (fully on-device) ---
        Tensor normed = VibeVoiceOps.RmsNormChannelsFirstGpu(backend, x, _normW!, batch, _dim, t, _normEps);

        Tensor mixed = cache is null
            ? _mixer.Forward(backend, normed, batch, t)
            : _mixer.ForwardStreaming(backend, normed, batch, t, cache, sampleIndices);
        normed.Dispose();

        // The depthwise mixer keeps T unchanged (stride=1). Sanity-check.
        if ((int)mixed.Shape[2] != t)
            throw new InvalidOperationException($"Mixer changed time dim: expected {t}, got {mixed.Shape[2]}.");

        Tensor mixedScaled = VibeVoiceOps.ChannelScaleGpu(backend, mixed, _gammaW!, batch, _dim, t);
        mixed.Dispose();

        Tensor afterMixer = new(x.Shape, DType.F32);
        backend.Add(afterMixer, x, mixedScaled);
        mixedScaled.Dispose();

        // --- ffn path ---
        Tensor normed2 = VibeVoiceOps.RmsNormChannelsFirstGpu(backend, afterMixer, _ffnNormW!, batch, _dim, t, _normEps);

        // Transpose [B, C, T] → [B, T, C] for channels-last FFN.
        Tensor cl = new(new TensorShape(batch, t, _dim), DType.F32);
        backend.Transpose2D(cl, normed2, _dim, t);
        normed2.Dispose();

        // FFN: Linear(C → 4C) → GELU → Linear(4C → C).
        Tensor up = WhisperOps.ProjectLinear(backend, cl, _ffn1W!, _ffn1B, batch, t, _dim, _ffnDim);
        cl.Dispose();
        Tensor act = new(up.Shape, DType.F32);
        backend.Gelu(act, up);
        up.Dispose();
        Tensor down = WhisperOps.ProjectLinear(backend, act, _ffn2W!, _ffn2B, batch, t, _ffnDim, _dim);
        act.Dispose();

        // Transpose back to channels-first.
        Tensor cf = new(new TensorShape(batch, _dim, t), DType.F32);
        backend.Transpose2D(cf, down, t, _dim);
        down.Dispose();

        Tensor cfScaled = VibeVoiceOps.ChannelScaleGpu(backend, cf, _ffnGammaW!, batch, _dim, t);
        cf.Dispose();

        Tensor result = new(afterMixer.Shape, DType.F32);
        backend.Add(result, afterMixer, cfScaled);
        afterMixer.Dispose();
        cfScaled.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_normW, _ffnNormW, _gammaW, _ffnGammaW, _ffn1W, _ffn1B, _ffn2W, _ffn2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
        foreach (Tensor t in _mixer.EnumerateWeights()) yield return t;
    }
}
