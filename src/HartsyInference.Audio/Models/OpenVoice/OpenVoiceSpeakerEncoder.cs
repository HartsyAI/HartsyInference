using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.OpenVoice;

/// <summary>OpenVoice V2 tone-color extractor — the converter's <c>ReferenceEncoder</c> that turns a reference
/// <b>linear spectrogram</b> into the speaker conditioning vector <c>g [1, gin, 1]</c> (gin 256). Faithful to
/// <c>myshell-ai/OpenVoiceV2/converter</c>: a LayerNorm over the spectrogram bins, a six-layer weight-normed
/// Conv2d stack (filters 32/32/64/64/128/128, kernel 3×3, stride 2×2, pad 1) with ReLU, a single-layer GRU
/// (hidden 128) over the flattened (channel-major × frequency) features, and a Linear to <c>gin</c> taken from
/// the GRU's final hidden state.
///
/// <para>Input is a linear spectrogram <c>[1, specChannels, T]</c> (channels-first, as
/// <see cref="LinearSpectrogram.Extract"/> produces); output is <c>[1, gin, 1]</c> so it drops straight into
/// the tone converter's <c>g</c> slot. Reuses <see cref="IBackend.Conv2D"/> / <see cref="IBackend.LayerNorm"/>,
/// the shared <see cref="WeightNorm"/> composition, the <see cref="Gru"/> layer, and
/// <see cref="WhisperOps.ProjectLinear"/>.</para></summary>
public sealed unsafe class OpenVoiceSpeakerEncoder : IDisposable
{
    private const float LayerNormEps = 1e-5f;

    private readonly OpenVoiceSpeakerConfig _cfg;
    private readonly Tensor?[] _convW;
    private readonly Tensor?[] _convB;
    private readonly int _gruInputDim;
    private readonly Gru _gru;

    private Tensor? _lnW, _lnB;         // LayerNorm over the spec bins
    private Tensor? _projW, _projB;     // Linear(gruHidden → gin)
    private int _disposed;

    public OpenVoiceSpeakerConfig Config => _cfg;

    public OpenVoiceSpeakerEncoder(OpenVoiceSpeakerConfig cfg)
    {
        _cfg = cfg;
        _convW = new Tensor[cfg.Channels.Count];
        _convB = new Tensor[cfg.Channels.Count];
        // The frequency axis is halved by every stride-2 conv; the GRU consumes channel-major × frequency.
        int freq = cfg.SpecChannels;
        for (int i = 0; i < cfg.Channels.Count; i++) freq = Down(freq);
        _gruInputDim = cfg.Channels[^1] * freq;
        _gru = new Gru(_gruInputDim, cfg.GruHidden);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "ref_enc")
    {
        _lnW = WhisperOps.EnsureF32(w[$"{prefix}.layernorm.weight"]);
        _lnB = WhisperOps.EnsureF32(w[$"{prefix}.layernorm.bias"]);
        for (int i = 0; i < _convW.Length; i++)
        {
            string cp = $"{prefix}.convs.{i}";
            _convW[i] = WeightNorm.Compose(w, cp);     // weight_g / weight_v → dense [out, in, 3, 3]
            _convB[i] = w.TryGetValue($"{cp}.bias", out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
        }
        _gru.LoadWeights(w, $"{prefix}.gru");
        _projW = WhisperOps.EnsureF32(w[$"{prefix}.proj.weight"]);
        _projB = w.TryGetValue($"{prefix}.proj.bias", out Tensor? pb) ? WhisperOps.EnsureF32(pb) : null;
    }

    /// <summary>Extracts the tone-color vector <c>g [1, gin, 1]</c> from a linear spectrogram
    /// <c>[1, specChannels, t]</c>.</summary>
    public Tensor Extract(IBackend backend, Tensor spec, int t)
    {
        if (_lnW is null) throw new InvalidOperationException("OpenVoiceSpeakerEncoder weights not loaded.");
        int specCh = _cfg.SpecChannels;

        // [1, spec, T] → [1, T, spec], LayerNorm over the spec bins, then view as NCHW [1, 1, T, spec].
        Tensor xTl = new(new TensorShape(1, t, specCh), DType.F32);
        backend.Transpose2D(xTl, spec, specCh, t);
        Tensor ln = new(xTl.Shape, DType.F32);
        backend.LayerNorm(ln, xTl, _lnW!, _lnB!, LayerNormEps);
        xTl.Dispose();

        Tensor x = ln.Reshape(new TensorShape(1, 1, t, specCh));
        int curH = t, curW = specCh;
        for (int i = 0; i < _convW.Length; i++)
        {
            int outC = _cfg.Channels[i];
            int outH = Down(curH), outW = Down(curW);
            Tensor conv = new(new TensorShape(1, outC, outH, outW), DType.F32);
            backend.Conv2D(conv, x, _convW[i]!, _convB[i], _cfg.Stride, _cfg.Stride, _cfg.Padding, _cfg.Padding);
            x.Dispose();
            backend.LeakyRelu(conv, conv, 0f);     // ReLU
            x = conv;
            curH = outH; curW = outW;
        }

        // torch: out.transpose(1,2).view(N, T', C*F) → feature index = channel-major × frequency.
        int lastC = _cfg.Channels[^1];
        int featDim = lastC * curW;
        Tensor seq = new(new TensorShape(1, curH, featDim), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* sp = (float*)seq.DataPointer;
        for (int h = 0; h < curH; h++)
            for (int c = 0; c < lastC; c++)
                for (int f = 0; f < curW; f++)
                    sp[(long)h * featDim + c * curW + f] = xp[((long)c * curH + h) * curW + f];
        x.Dispose();

        Tensor last = _gru.LastHidden(backend, seq, 1, curH);     // [1, gruHidden]
        seq.Dispose();
        Tensor last3 = last.Reshape(new TensorShape(1, 1, _cfg.GruHidden));
        Tensor proj = WhisperOps.ProjectLinear(backend, last3, _projW!, _projB, 1, 1, _cfg.GruHidden, _cfg.Gin);
        last.Dispose();
        return proj.Reshape(new TensorShape(1, _cfg.Gin, 1));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_lnW, _lnB, _projW, _projB];
        foreach (Tensor? x in top) if (x is not null) yield return x;
        for (int i = 0; i < _convW.Length; i++)
        {
            if (_convW[i] is not null) yield return _convW[i]!;
            if (_convB[i] is not null) yield return _convB[i]!;
        }
        foreach (Tensor t in _gru.EnumerateWeights()) yield return t;
    }

    private int Down(int dim) => (dim + 2 * _cfg.Padding - _cfg.KernelSize) / _cfg.Stride + 1;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
