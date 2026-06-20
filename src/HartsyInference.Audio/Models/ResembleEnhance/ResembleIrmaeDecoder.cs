using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ResembleEnhance;

/// <summary>IRMAE latent decoder (resemble-enhance LCFM): latent <c>[1, latent, T]</c> → mel <c>[1, mels, T]</c>
/// via <c>Conv1d(latent→hidden) → N dilated ResBlocks (GroupNorm/GELU) → Conv1d(hidden→mels) → head</c>. No
/// temporal resampling. (The IRMAE encoder is training-only.) Reuses <c>IBackend</c> Conv1d/GroupNorm/GELU.</summary>
public sealed unsafe class ResembleIrmaeDecoder
{
    private readonly ResembleEnhanceConfig _cfg;
    private readonly int _hidden, _mels, _latent;
    private Tensor? _inW, _inB, _midW, _midB, _h1W, _h1B, _h2W, _h2B;
    private readonly Res[] _blocks;

    public ResembleIrmaeDecoder(ResembleEnhanceConfig cfg)
    {
        _cfg = cfg; _hidden = cfg.AeHidden; _mels = cfg.NMels; _latent = cfg.LatentDim;
        _blocks = new Res[cfg.AeResBlocks];
        for (int i = 0; i < _blocks.Length; i++) _blocks[i] = new Res(_hidden, cfg.AeDilations[i], cfg.NormEps);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "lcfm.ae.decoder")
    {
        _inW = WhisperOps.EnsureF32(w[$"{prefix}.in_conv.weight"]); _inB = WhisperOps.EnsureF32(w[$"{prefix}.in_conv.bias"]);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"{prefix}.res.{i}");
        _midW = WhisperOps.EnsureF32(w[$"{prefix}.out_conv.weight"]); _midB = WhisperOps.EnsureF32(w[$"{prefix}.out_conv.bias"]);
        _h1W = WhisperOps.EnsureF32(w[$"{prefix}.head.0.weight"]); _h1B = WhisperOps.EnsureF32(w[$"{prefix}.head.0.bias"]);
        _h2W = WhisperOps.EnsureF32(w[$"{prefix}.head.2.weight"]); _h2B = WhisperOps.EnsureF32(w[$"{prefix}.head.2.bias"]);
    }

    public Tensor Decode(IBackend backend, Tensor latent, int t)
    {
        Tensor x = new(new TensorShape(1, _hidden, t), DType.F32);
        backend.Conv1d(x, latent, _inW!, _inB, 1, 1, 1, 1, 1);
        foreach (Res r in _blocks) { Tensor n = r.Forward(backend, x, t); x.Dispose(); x = n; }
        Tensor mel = new(new TensorShape(1, _mels, t), DType.F32);
        backend.Conv1d(mel, x, _midW!, _midB, 1, 0, 0, 1, 1); x.Dispose();
        // head: Conv1d(mels→hidden,k3) → GELU → Conv1d(hidden→mels,k1).
        Tensor h = new(new TensorShape(1, _hidden, t), DType.F32);
        backend.Conv1d(h, mel, _h1W!, _h1B, 1, 1, 1, 1, 1);
        Tensor g = new(h.Shape, DType.F32); backend.Gelu(g, h); h.Dispose();
        Tensor outMel = new(new TensorShape(1, _mels, t), DType.F32);
        backend.Conv1d(outMel, g, _h2W!, _h2B, 1, 0, 0, 1, 1); g.Dispose(); mel.Dispose();
        return outMel;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_inW, _inB, _midW, _midB, _h1W, _h1B, _h2W, _h2B];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (Res r in _blocks) foreach (Tensor t in r.EnumerateWeights()) yield return t;
    }

    private sealed class Res
    {
        private readonly int _ch, _dilation;
        private readonly float _eps;
        private Tensor? _n1W, _n1B, _c1W, _c1B, _n2W, _n2B, _c2W, _c2B;

        public Res(int ch, int dilation, float eps) { _ch = ch; _dilation = dilation; _eps = eps; }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _n1W = WhisperOps.EnsureF32(w[$"{p}.norm1.weight"]); _n1B = WhisperOps.EnsureF32(w[$"{p}.norm1.bias"]);
            _c1W = WhisperOps.EnsureF32(w[$"{p}.conv1.weight"]); _c1B = WhisperOps.EnsureF32(w[$"{p}.conv1.bias"]);
            _n2W = WhisperOps.EnsureF32(w[$"{p}.norm2.weight"]); _n2B = WhisperOps.EnsureF32(w[$"{p}.norm2.bias"]);
            _c2W = WhisperOps.EnsureF32(w[$"{p}.conv2.weight"]); _c2B = WhisperOps.EnsureF32(w[$"{p}.conv2.bias"]);
        }

        public Tensor Forward(IBackend backend, Tensor x, int t)
        {
            int groups = Math.Max(1, _ch / 16);
            Tensor n1 = new(x.Shape, DType.F32); backend.GroupNorm(n1, x, _n1W!, _n1B!, groups, _eps);
            Tensor g1 = new(x.Shape, DType.F32); backend.Gelu(g1, n1); n1.Dispose();
            int pad = _dilation;
            Tensor c1 = new(x.Shape, DType.F32); backend.Conv1d(c1, g1, _c1W!, _c1B, 1, pad, pad, _dilation, 1); g1.Dispose();
            Tensor n2 = new(x.Shape, DType.F32); backend.GroupNorm(n2, c1, _n2W!, _n2B!, groups, _eps); c1.Dispose();
            Tensor g2 = new(x.Shape, DType.F32); backend.Gelu(g2, n2); n2.Dispose();
            Tensor c2 = new(x.Shape, DType.F32); backend.Conv1d(c2, g2, _c2W!, _c2B, 1, 1, 1, 1, 1); g2.Dispose();
            Tensor outT = new(x.Shape, DType.F32); backend.Add(outT, x, c2); c2.Dispose();
            return outT;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] a = [_n1W, _n1B, _c1W, _c1B, _n2W, _n2B, _c2W, _c2B];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }
}
