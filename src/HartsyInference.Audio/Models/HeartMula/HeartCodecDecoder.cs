using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.HeartMula;

/// <summary>HeartCodec decoder: 48 kHz / 12.5 Hz, 8-codebook RVQ (codebook_size 8192, codebook_dim 32). The
/// integer codes are dequantized by a per-book learned codebook lookup that is summed into a continuous
/// latent (each book has its own <c>out_proj</c> 1×1 conv back to the codec dim, like DAC). That latent
/// conditions a <b>flow-matching</b> reconstruction head (SQ-Codec style): a multi-scale causal conv
/// velocity network is integrated by the reused OT-CFM Euler solver (<see cref="ConditionalCfm"/>) from
/// noise to the codec feature, then a transposed-conv upsampling stack (<c>downsample [3,4,4,4,5]</c>
/// reversed, dim 512, causal) lifts it to the 48 kHz waveform. See
/// <c>docs/Research/HEARTMULA_ARCHITECTURE.md</c>.
///
/// <para><b>Reuse:</b> the codebook-lookup + per-book out_proj mirror <c>DacResidualVectorQuantizer</c>; the
/// velocity net is an <see cref="ICfmEstimator"/> driven by the existing solver; the upsample stack uses
/// <c>backend.ConvTranspose1d</c> with causal trailing-pad removal (<c>padRight = K - stride</c>).</para>
///
/// <para><b>Keys</b> (HeartCodec checkpoint): <c>quantizer.quantizers.{q}.codebook.weight</c> /
/// <c>.out_proj.{weight,bias}</c>; <c>flow.net.*</c> (the CFM velocity net, WaveNet-style); <c>decoder.in.*</c>,
/// <c>decoder.ups.{i}.*</c>, <c>decoder.out.*</c> (the transposed-conv upsampler). Parameterized from
/// <see cref="HeartMulaConfig"/>.</para></summary>
public sealed unsafe class HeartCodecDecoder : IDisposable
{
    // downsample [3,4,4,4,5] reversed gives the upsample strides; product 960 = 48000 / (12.5 * 4)
    // upsampled per the flow-feature rate (the flow net runs at 4× the 12.5 Hz code rate).
    private static readonly int[] UpStrides = [5, 4, 4, 4, 3];

    private readonly HeartMulaConfig _cfg;
    private readonly int _nq, _codebookSize, _codebookDim, _dim;
    private readonly int _cfmNfe;
    private readonly float _cfmCfg;

    private readonly Tensor?[] _codebooks;        // [q] → [codebook_size, codebook_dim]
    private readonly Tensor?[] _outProjW;         // [q] → 1×1 conv codebook_dim → dim
    private readonly Tensor?[] _outProjB;

    private readonly HeartCodecCfmEstimator _flow;
    private readonly ConditionalCfm _cfm;

    // Transposed-conv upsampler.
    private Tensor? _inW, _inB, _outW, _outB;
    private readonly Tensor?[] _upW, _upB;
    private int _disposed;

    public HeartCodecDecoder(HeartMulaConfig cfg, int dim = 512, int cfmNfe = 10, float cfmCfg = 0f)
    {
        _cfg = cfg;
        _nq = cfg.CodecNumQuantizers;
        _codebookSize = cfg.CodecCodebookSize;
        _codebookDim = cfg.CodecCodebookDim;
        _dim = dim;
        _cfmNfe = cfmNfe;
        _cfmCfg = cfmCfg;
        _codebooks = new Tensor?[_nq];
        _outProjW = new Tensor?[_nq];
        _outProjB = new Tensor?[_nq];
        _upW = new Tensor?[UpStrides.Length];
        _upB = new Tensor?[UpStrides.Length];
        _flow = new HeartCodecCfmEstimator(dim);
        _cfm = new ConditionalCfm(_flow, dim);
    }

    public int SampleRate => _cfg.Lm.SampleRate;

    /// <summary>Per-frame upsampling ratio from the flow feature to the waveform (∏ of the transposed strides).</summary>
    public int HopSize { get { int h = 1; foreach (int s in UpStrides) h *= s; return h; } }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        for (int q = 0; q < _nq; q++)
        {
            string qp = $"{p}quantizer.quantizers.{q}";
            _codebooks[q] = WhisperOps.EnsureF32(w[$"{qp}.codebook.weight"]);
            _outProjW[q] = WhisperOps.EnsureF32(w[$"{qp}.out_proj.weight"]);
            _outProjB[q] = Bias(w, $"{qp}.out_proj.bias");
        }

        _flow.LoadWeights(w, $"{p}flow.net");

        _inW = WhisperOps.EnsureF32(w[$"{p}decoder.in.weight"]);
        _inB = Bias(w, $"{p}decoder.in.bias");
        for (int i = 0; i < UpStrides.Length; i++)
        {
            _upW[i] = WhisperOps.EnsureF32(w[$"{p}decoder.ups.{i}.weight"]);
            _upB[i] = Bias(w, $"{p}decoder.ups.{i}.bias");
        }
        _outW = WhisperOps.EnsureF32(w[$"{p}decoder.out.weight"]);
        _outB = Bias(w, $"{p}decoder.out.bias");
    }

    /// <summary>Decodes an 8-codebook grid <c>[NumCodebooks, T]</c> into a 48 kHz mono waveform. RVQ dequant
    /// → flow-matching head (CFM-solved codec feature) → transposed-conv upsampler.</summary>
    public float[] Decode(IBackend backend, int[,] codes, int seed)
    {
        ThrowIfDisposed();
        if (_codebooks[0] is null) throw new InvalidOperationException("HeartCodecDecoder weights not loaded.");
        int nb = codes.GetLength(0);
        int t = codes.GetLength(1);
        if (nb != _nq) throw new ArgumentException($"codes has {nb} books, expected {_nq}.", nameof(codes));
        if (t <= 0) return [];

        // RVQ dequant → continuous latent [1, dim, T], the CFM conditioning (mu).
        Tensor mu = Dequantize(backend, codes, t);

        // Flow-matching reconstruction head: integrate noise → codec feature conditioned on mu.
        Tensor spk = new(new TensorShape(1, _dim, t), DType.F32);     // unused conditioning (zeros)
        Tensor cond = new(new TensorShape(1, _dim, t), DType.F32);
        Tensor feat = _cfm.Solve(backend, mu, spk, cond, _cfmNfe, _cfmCfg, seed);
        mu.Dispose(); spk.Dispose(); cond.Dispose();

        // Transposed-conv upsampler → waveform.
        float[] wave = Upsample(backend, feat, t);
        feat.Dispose();
        return wave;
    }

    /// <summary>Per-book codebook lookup summed into the codec latent. Each book's <c>codebook_dim</c>
    /// vector is mapped to the codec dim by its <c>out_proj</c> 1×1 conv and accumulated (RVQ residual sum).</summary>
    private Tensor Dequantize(IBackend backend, int[,] codes, int t)
    {
        Tensor latent = new(new TensorShape(1, _dim, t), DType.F32);
        float* lp = (float*)latent.DataPointer;
        long n = (long)_dim * t;
        for (long i = 0; i < n; i++) lp[i] = 0f;

        for (int q = 0; q < _nq; q++)
        {
            Tensor quant = new(new TensorShape(1, _codebookDim, t), DType.F32);
            float* qp = (float*)quant.DataPointer;
            float* cb = (float*)_codebooks[q]!.DataPointer;
            for (int ti = 0; ti < t; ti++)
            {
                int idx = codes[q, ti];
                if ((uint)idx >= (uint)_codebookSize)
                    throw new ArgumentOutOfRangeException(nameof(codes), idx, $"code at book {q}, t {ti} out of range [0,{_codebookSize}).");
                int rowBase = idx * _codebookDim;
                for (int d = 0; d < _codebookDim; d++) qp[(long)d * t + ti] = cb[rowBase + d];
            }

            Tensor proj = new(new TensorShape(1, _dim, t), DType.F32);
            backend.Conv1d(proj, quant, _outProjW[q]!, _outProjB[q], 1, 0, 0, 1, 1);
            quant.Dispose();

            float* pp = (float*)proj.DataPointer;
            for (long i = 0; i < n; i++) lp[i] += pp[i];
            proj.Dispose();
        }
        return latent;
    }

    private float[] Upsample(IBackend backend, Tensor feat, int t)
    {
        // in conv: dim → dim (causal, kernel from weight).
        int inK = (int)_inW!.Shape[2];
        Tensor cur = new(new TensorShape(1, _dim, t), DType.F32);
        backend.Conv1d(cur, feat, _inW!, _inB, 1, inK - 1, 0, 1, 1);

        int curLen = t;
        for (int i = 0; i < UpStrides.Length; i++)
        {
            int stride = UpStrides[i];
            int k = (int)_upW[i]!.Shape[2];
            int outCh = (int)_upW[i]!.Shape[1];     // ConvTranspose weight [C_in, C_out, K]
            // Causal transposed conv: padRight = K - stride removes trailing pad → outLen = curLen * stride.
            int outLen = (curLen - 1) * stride + (k - 1) + 1 - (k - stride);
            Tensor up = new(new TensorShape(1, outCh, outLen), DType.F32);
            backend.ConvTranspose1d(up, cur, _upW[i]!, _upB[i], stride, 0, k - stride, 1, 1);
            cur.Dispose();
            // SiLU between upsampling stages.
            backend.Silu(up, up);
            cur = up;
            curLen = outLen;
        }

        // out conv: dim → 1 (mono), causal.
        int outK = (int)_outW!.Shape[2];
        Tensor wav = new(new TensorShape(1, 1, curLen), DType.F32);
        backend.Conv1d(wav, cur, _outW!, _outB, 1, outK - 1, 0, 1, 1);
        cur.Dispose();

        float[] result = new float[curLen];
        float* wp = (float*)wav.DataPointer;
        for (int i = 0; i < curLen; i++) result[i] = MathF.Tanh(wp[i]);
        wav.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int q = 0; q < _nq; q++)
        {
            Tensor?[] qs = [_codebooks[q], _outProjW[q], _outProjB[q]];
            foreach (Tensor? t in qs) if (t is not null) yield return t;
        }
        foreach (Tensor t in _flow.EnumerateWeights()) yield return t;
        Tensor?[] own = [_inW, _inB, _outW, _outB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        for (int i = 0; i < UpStrides.Length; i++)
        {
            if (_upW[i] is not null) yield return _upW[i]!;
            if (_upB[i] is not null) yield return _upB[i]!;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(HeartCodecDecoder));
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
}
