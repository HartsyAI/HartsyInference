using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ResembleEnhance;

/// <summary>IRMAE latent autoencoder (upstream <c>enhancer/lcfm/irmae.py</c>). Encoder: <c>Conv1d(128→1024, k3)</c>
/// → 4 ResBlocks → 4 bias-free 1×1 IRM convs (<c>1024→64</c>, then <c>64→64</c> ×3) → Tanh, producing the latent
/// the CFM prior starts from at inference (ψ0 mixing). Decoder: <c>Conv1d(64→1024, k3)</c> → 4 ResBlocks →
/// <c>Conv1d(1024→160, k1)</c>, whose 160-channel output (mel + vocoder extra dim) feeds UnivNet directly. Each
/// ResBlock is one residual around four (GroupNorm(32) → GELU → weight-norm Conv1d k3, dilations 1/2/4/8) stages.
/// The training-only reconstruction <c>head</c> and the <c>estimator</c> running stats are consumed but unused.</summary>
public sealed unsafe class ResembleIrmae
{
    private readonly int _hidden, _melDim, _latent, _outputDim;
    private Tensor? _encInW, _encInB, _decInW, _decInB, _decOutW, _decOutB;
    private readonly Tensor?[] _irmW;
    private readonly ResBlock[] _encBlocks;
    private readonly ResBlock[] _decBlocks;

    public ResembleIrmae(ResembleEnhanceConfig cfg)
    {
        _hidden = cfg.AeHidden;
        _melDim = cfg.NMels;
        _latent = cfg.LatentDim;
        _outputDim = cfg.NMels + cfg.VocoderExtraDim;
        _irmW = new Tensor?[cfg.NumIrms];
        _encBlocks = new ResBlock[cfg.AeResBlocks];
        _decBlocks = new ResBlock[cfg.AeResBlocks];
        for (int i = 0; i < cfg.AeResBlocks; i++)
        {
            _encBlocks[i] = new ResBlock(cfg.AeDilations, cfg.IrmaeGroupNorm);
            _decBlocks[i] = new ResBlock(cfg.AeDilations, cfg.IrmaeGroupNorm);
        }
    }

    public void LoadWeights(ResembleWeightReader r, string prefix = "lcfm.ae")
    {
        _encInW = r.F32($"{prefix}.encoder.0.weight");
        _encInB = r.F32($"{prefix}.encoder.0.bias");
        for (int i = 0; i < _encBlocks.Length; i++)
        {
            _encBlocks[i].LoadWeights(r, $"{prefix}.encoder.{i + 1}");
        }
        for (int i = 0; i < _irmW.Length; i++)
        {
            _irmW[i] = r.F32($"{prefix}.encoder.{_encBlocks.Length + 1 + i}.weight");
        }
        _decInW = r.F32($"{prefix}.decoder.0.weight");
        _decInB = r.F32($"{prefix}.decoder.0.bias");
        for (int i = 0; i < _decBlocks.Length; i++)
        {
            _decBlocks[i].LoadWeights(r, $"{prefix}.decoder.{i + 1}");
        }
        _decOutW = r.F32($"{prefix}.decoder.{_decBlocks.Length + 1}.weight");
        _decOutB = r.F32($"{prefix}.decoder.{_decBlocks.Length + 1}.bias");
        // Training-only reconstruction head and latent-stats estimator: present in the checkpoint, no
        // inference role — consumed so the strict key check stays exhaustive.
        r.MarkUnused(
            $"{prefix}.head.0.weight", $"{prefix}.head.0.bias",
            $"{prefix}.head.2.weight", $"{prefix}.head.2.bias",
            $"{prefix}.estimator.running_mean_unsafe", $"{prefix}.estimator.running_var_unsafe");
    }

    /// <summary>Encodes a normalized mel <c>[1, mels, T]</c> to the Tanh-bounded latent <c>[1, latent, T]</c>.</summary>
    public Tensor Encode(IBackend backend, Tensor mel)
    {
        int t = (int)mel.Shape[2];
        Tensor x = new(new TensorShape(1, _hidden, t), DType.F32);
        backend.Conv1d(x, mel, _encInW!, _encInB, 1, 1, 1, 1, 1);
        foreach (ResBlock b in _encBlocks)
        {
            Tensor n = b.Forward(backend, x);
            x.Dispose();
            x = n;
        }
        Tensor z = new(new TensorShape(1, _latent, t), DType.F32);
        backend.Conv1d(z, x, _irmW[0]!, null, 1, 0, 0, 1, 1);
        x.Dispose();
        for (int i = 1; i < _irmW.Length; i++)
        {
            Tensor n = new(z.Shape, DType.F32);
            backend.Conv1d(n, z, _irmW[i]!, null, 1, 0, 0, 1, 1);
            z.Dispose();
            z = n;
        }
        Tensor bounded = new(z.Shape, DType.F32);
        backend.Tanh(bounded, z);
        z.Dispose();
        return bounded;
    }

    /// <summary>Decodes a latent <c>[1, latent, T]</c> to the vocoder input <c>[1, mels+extra, T]</c>.</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        int t = (int)latent.Shape[2];
        Tensor x = new(new TensorShape(1, _hidden, t), DType.F32);
        backend.Conv1d(x, latent, _decInW!, _decInB, 1, 1, 1, 1, 1);
        foreach (ResBlock b in _decBlocks)
        {
            Tensor n = b.Forward(backend, x);
            x.Dispose();
            x = n;
        }
        Tensor outT = new(new TensorShape(1, _outputDim, t), DType.F32);
        backend.Conv1d(outT, x, _decOutW!, _decOutB, 1, 0, 0, 1, 1);
        x.Dispose();
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_encInW, _encInB, _decInW, _decInB, _decOutW, _decOutB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (Tensor? t in _irmW) if (t is not null) yield return t;
        foreach (ResBlock b in _encBlocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (ResBlock b in _decBlocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Upstream <c>irmae.ResBlock</c>: ONE residual around four GroupNorm(32) → GELU → weight-norm
    /// Conv1d(k3, same-pad) stages at dilations 1/2/4/8. Checkpoint Sequential indices per stage i:
    /// <c>3i</c>=GN, <c>3i+2</c>=Conv (<c>weight_g</c>/<c>weight_v</c>).</summary>
    private sealed class ResBlock
    {
        private readonly int _groups;
        private readonly IReadOnlyList<int> _dilations;
        private readonly Tensor?[] _nW, _nB, _cW, _cB;

        public ResBlock(IReadOnlyList<int> dilations, int groups)
        {
            _dilations = dilations;
            _groups = groups;
            _nW = new Tensor?[dilations.Count];
            _nB = new Tensor?[dilations.Count];
            _cW = new Tensor?[dilations.Count];
            _cB = new Tensor?[dilations.Count];
        }

        public void LoadWeights(ResembleWeightReader r, string p)
        {
            for (int i = 0; i < _dilations.Count; i++)
            {
                _nW[i] = r.F32($"{p}.{3 * i}.weight");
                _nB[i] = r.F32($"{p}.{3 * i}.bias");
                _cW[i] = r.WeightNormF32($"{p}.{3 * i + 2}");
                _cB[i] = r.F32($"{p}.{3 * i + 2}.bias");
            }
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            Tensor h = x;
            bool owned = false;
            for (int i = 0; i < _dilations.Count; i++)
            {
                Tensor n = new(x.Shape, DType.F32);
                backend.GroupNorm(n, h, _nW[i]!, _nB[i]!, _groups, 1e-5f);
                if (owned) h.Dispose();
                Tensor g = new(x.Shape, DType.F32);
                backend.GeluErf(g, n);
                n.Dispose();
                int pad = _dilations[i];
                Tensor c = new(x.Shape, DType.F32);
                backend.Conv1d(c, g, _cW[i]!, _cB[i], 1, pad, pad, _dilations[i], 1);
                g.Dispose();
                h = c;
                owned = true;
            }
            Tensor outT = new(x.Shape, DType.F32);
            backend.Add(outT, x, h);
            h.Dispose();
            return outT;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[][] all = [_nW, _nB, _cW, _cB];
            foreach (Tensor?[] arr in all) foreach (Tensor? t in arr) if (t is not null) yield return t;
        }
    }
}
