using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Rvc;

/// <summary>RMVPE (Robust Model for Vocal Pitch Estimation) — the per-frame f0 extractor RVC v2 uses, matching
/// the upstream <c>E2E(4, 1, (2,2))</c> (RVC-WebUI <c>rmvpe.py</c>). A DeepUNet over a 128-mel spectrogram
/// (BatchNorm stem → 5 stride-2 <see cref="ResEncoderBlock"/> stages, each 4× <see cref="ConvBlockRes"/> then
/// AvgPool2d; 4 intermediate ResEncoderBlocks; 5 <see cref="ResDecoderBlock"/> stages with ConvTranspose2d
/// upsample + skip concat) → a 16→3 conv → flatten to 384 → a single bidirectional GRU (hidden 256) → Linear
/// (512→360) → sigmoid posterior over pitch bins. The posterior is decoded to Hz via a local weighted average.
///
/// <para>Real-weight key layout (<c>rmvpe.pt</c>): <c>unet.encoder.bn.*</c>, <c>unet.encoder.layers.N.conv.M.*</c>
/// (ConvBlockRes <c>conv.0/1/3/4</c> + optional <c>shortcut</c>), <c>unet.intermediate.layers.N.*</c>,
/// <c>unet.decoder.layers.N.{conv1,conv2.M}.*</c>, <c>cnn.*</c>, <c>fc.0.gru.*_l0[_reverse]</c>, <c>fc.1.*</c>.
/// All convs are bias-free with BatchNorm2d (eps 1e-5, folded to per-channel affine at load).</para></summary>
public sealed unsafe class RvcRmvpe : IDisposable
{
    private const int InterBlocks = 4;       // E2E inter_layers
    private const int Stages = 5;            // en_de_layers
    private const int NBlocks = 4;           // ConvBlockRes per ResEncoderBlock
    private const int GruIn = 384;           // 3 * 128
    private const int GruHidden = 256;

    private readonly RvcRmvpeConfig _cfg;
    private readonly MelSpectrogramExtractor _mel;

    private BatchNorm2d? _encBn;
    private ResEncoderBlock[] _encoder = [];
    private ResEncoderBlock[] _intermediate = [];
    private ResDecoderBlock[] _decoder = [];
    private Tensor? _cnnW, _cnnB;
    private Gru? _gruFwd, _gruRev;
    private Tensor? _fcW, _fcB;          // Linear(512 → 360)
    private readonly float[] _centsMapping;
    private int _disposed;

    public RvcRmvpeConfig Config => _cfg;

    public RvcRmvpe(RvcRmvpeConfig cfg)
    {
        _cfg = cfg;
        _mel = new MelSpectrogramExtractor(cfg.MelConfig());
        _centsMapping = new float[cfg.NumBins];
        for (int k = 0; k < cfg.NumBins; k++)
            _centsMapping[k] = cfg.FirstFreqHz * MathF.Pow(2f, k * cfg.CentsPerBin / 1200f);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = string.IsNullOrEmpty(prefix) ? "" : prefix + ".";

        _encBn = BatchNorm2d.Load(w, $"{p}unet.encoder.bn");
        // Encoder: 5 stages, channels 1→16→32→64→128→256 (out doubles each stage), AvgPool after each.
        _encoder = new ResEncoderBlock[Stages];
        int inC = 1, outC = 16;
        for (int i = 0; i < Stages; i++)
        {
            _encoder[i] = ResEncoderBlock.Load(w, $"{p}unet.encoder.layers.{i}", inC, outC, NBlocks, pool: true);
            inC = outC; outC *= 2;
        }
        // Intermediate: (256→512) then 3× (512→512), no pooling.
        int interIn = outC / 2;       // 256
        int interOut = outC;          // 512
        _intermediate = new ResEncoderBlock[InterBlocks];
        for (int i = 0; i < InterBlocks; i++)
        {
            _intermediate[i] = ResEncoderBlock.Load(w, $"{p}unet.intermediate.layers.{i}",
                i == 0 ? interIn : interOut, interOut, NBlocks, pool: false);
        }
        // Decoder: 5 stages, channels 512→256→128→64→32→16, ConvTranspose2d upsample + skip concat.
        _decoder = new ResDecoderBlock[Stages];
        int decIn = interOut;
        for (int i = 0; i < Stages; i++)
        {
            int decOut = decIn / 2;
            _decoder[i] = ResDecoderBlock.Load(w, $"{p}unet.decoder.layers.{i}", decIn, decOut, NBlocks);
            decIn = decOut;
        }

        _cnnW = WhisperOps.EnsureF32(w[$"{p}cnn.weight"]);
        _cnnB = TryGet(w, $"{p}cnn.bias");

        // Bidirectional GRU (single layer): forward (l0) + reverse (l0_reverse).
        _gruFwd = new Gru(GruIn, GruHidden);
        _gruFwd.LoadWeights(w, $"{p}fc.0.gru");
        _gruRev = new Gru(GruIn, GruHidden);
        _gruRev.LoadWeights(RenameReverse(w, $"{p}fc.0.gru"), "g");

        _fcW = WhisperOps.EnsureF32(w[$"{p}fc.1.weight"]);
        _fcB = TryGet(w, $"{p}fc.1.bias");
    }

    /// <summary>Optional per-stage activation hook for parity debugging (key = stage name). Not used in production.</summary>
    public Action<string, Tensor>? DebugHook { get; set; }

    /// <summary>Runs the E2E network on a precomputed mel <c>[1, melBins, frames]</c> (channels-first) and returns
    /// the <c>[1, frames, 360]</c> sigmoid pitch posterior. Exposed for parity testing against the mel→hidden oracle;
    /// <see cref="ExtractF0"/> wraps the mel front-end + Hz decode around it.</summary>
    public Tensor Hidden(IBackend backend, Tensor mel, int melBins, int frames)
    {
        if (_encBn is null) throw new InvalidOperationException("RvcRmvpe weights not loaded.");

        // E2E.forward: mel [1,melBins,T] -> transpose -> [1,1,T,melBins].
        Tensor x = new(new TensorShape(1, 1, frames, melBins), DType.F32);
        float* mp = (float*)mel.DataPointer;
        float* xp = (float*)x.DataPointer;
        for (int m = 0; m < melBins; m++)
            for (int t = 0; t < frames; t++)
                xp[(long)t * melBins + m] = mp[(long)m * frames + t];

        // Encoder BatchNorm on 1 channel, then stages.
        _encBn!.Apply(x, 1, frames * melBins);
        DebugHook?.Invoke("enc_bn", x);

        Tensor[] skips = new Tensor[Stages];
        int h = frames, wDim = melBins;
        for (int i = 0; i < Stages; i++)
        {
            (Tensor pre, Tensor pooled) = _encoder[i].Forward(backend, x, h, wDim);
            x.Dispose();
            skips[i] = pre;
            x = pooled;
            h /= 2; wDim /= 2;
            DebugHook?.Invoke($"enc_{i}", x);
        }
        foreach (ResEncoderBlock blk in _intermediate)
        {
            (Tensor outT, _) = blk.Forward(backend, x, h, wDim);
            x.Dispose();
            x = outT;
        }
        DebugHook?.Invoke("inter", x);
        for (int i = 0; i < Stages; i++)
        {
            Tensor skip = skips[Stages - 1 - i];
            Tensor next = _decoder[i].Forward(backend, x, h, wDim, skip);
            x.Dispose();
            skip.Dispose();
            x = next;
            h *= 2; wDim *= 2;
            DebugHook?.Invoke($"dec_{i}", x);
        }

        // cnn: Conv2d(16 -> 3, 3x3, pad 1).
        int cnnOut = (int)_cnnW!.Shape[0];
        Tensor c = new(new TensorShape(1, cnnOut, h, wDim), DType.F32);
        backend.Conv2D(c, x, _cnnW!, _cnnB, 1, 1, 1, 1);
        x.Dispose();
        DebugHook?.Invoke("cnn", c);

        // Flatten [1,cnnOut,T,W] -> [1,T,cnnOut*W] (transpose(1,2).flatten(-2)).
        int feat = cnnOut * wDim;
        Tensor flat = new(new TensorShape(1, h, feat), DType.F32);
        float* cp = (float*)c.DataPointer;
        float* fp = (float*)flat.DataPointer;
        for (int t = 0; t < h; t++)
            for (int ch = 0; ch < cnnOut; ch++)
                for (int wv = 0; wv < wDim; wv++)
                    fp[((long)t * feat) + (long)ch * wDim + wv] = cp[(((long)ch * h) + t) * wDim + wv];
        c.Dispose();
        DebugHook?.Invoke("flat", flat);

        // Bidirectional GRU -> [1,T,512].
        Tensor fwd = _gruFwd!.Forward(backend, flat, 1, h);
        Tensor revIn = ReverseTime(flat, h, feat);
        flat.Dispose();
        Tensor revOutR = _gruRev!.Forward(backend, revIn, 1, h);
        revIn.Dispose();
        Tensor rev = ReverseTime(revOutR, h, GruHidden);
        revOutR.Dispose();
        Tensor bigru = new(new TensorShape(1, h, 2 * GruHidden), DType.F32);
        float* bp = (float*)bigru.DataPointer;
        float* ffp = (float*)fwd.DataPointer;
        float* rrp = (float*)rev.DataPointer;
        for (int t = 0; t < h; t++)
        {
            for (int k = 0; k < GruHidden; k++) bp[(long)t * 2 * GruHidden + k] = ffp[(long)t * GruHidden + k];
            for (int k = 0; k < GruHidden; k++) bp[(long)t * 2 * GruHidden + GruHidden + k] = rrp[(long)t * GruHidden + k];
        }
        fwd.Dispose(); rev.Dispose();
        DebugHook?.Invoke("gru", bigru);

        // fc: Linear(512 -> 360) + sigmoid.
        Tensor logits = WhisperOps.ProjectLinear(backend, bigru, _fcW!, _fcB, 1, h, 2 * GruHidden, _cfg.NumBins);
        bigru.Dispose();
        float* lp = (float*)logits.DataPointer;
        for (long i = 0; i < logits.ElementCount; i++) lp[i] = 1f / (1f + MathF.Exp(-lp[i]));
        DebugHook?.Invoke("hidden", logits);
        return logits;
    }

    /// <summary>Extracts per-frame f0 in Hz (0 = unvoiced) from 16 kHz mono PCM.</summary>
    public float[] ExtractF0(IBackend backend, float[] pcm16k)
    {
        float[,] mel2d = _mel.Compute(pcm16k);
        int melBins = _cfg.MelBins;
        int frames = mel2d.GetLength(1);
        Tensor mel = new(new TensorShape(1, melBins, frames), DType.F32);
        float* mp = (float*)mel.DataPointer;
        for (int m = 0; m < melBins; m++)
            for (int t = 0; t < frames; t++) mp[(long)m * frames + t] = mel2d[m, t];

        Tensor hidden = Hidden(backend, mel, melBins, frames);
        mel.Dispose();
        float* lp = (float*)hidden.DataPointer;
        float[] f0 = DecodeF0(lp, frames);
        hidden.Dispose();
        return f0;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_encBn is not null) foreach (Tensor t in _encBn.Weights()) yield return t;
        foreach (ResEncoderBlock e in _encoder) foreach (Tensor t in e.Weights()) yield return t;
        foreach (ResEncoderBlock e in _intermediate) foreach (Tensor t in e.Weights()) yield return t;
        foreach (ResDecoderBlock d in _decoder) foreach (Tensor t in d.Weights()) yield return t;
        if (_cnnW is not null) yield return _cnnW;
        if (_cnnB is not null) yield return _cnnB;
        if (_gruFwd is not null) foreach (Tensor t in _gruFwd.EnumerateWeights()) yield return t;
        if (_gruRev is not null) foreach (Tensor t in _gruRev.EnumerateWeights()) yield return t;
        if (_fcW is not null) yield return _fcW;
        if (_fcB is not null) yield return _fcB;
    }

    private static Tensor ReverseTime(Tensor x, int t, int dim)
    {
        Tensor r = new(new TensorShape(1, t, dim), DType.F32);
        float* sp = (float*)x.DataPointer;
        float* dp = (float*)r.DataPointer;
        for (int i = 0; i < t; i++)
            for (int k = 0; k < dim; k++) dp[(long)i * dim + k] = sp[(long)(t - 1 - i) * dim + k];
        return r;
    }

    private static Dictionary<string, Tensor> RenameReverse(IReadOnlyDictionary<string, Tensor> w, string gruPrefix)
    {
        // Map the *_reverse weights of the bidirectional GRU onto plain l0 keys so a unidirectional Gru loads them.
        Dictionary<string, Tensor> d = new();
        foreach (string suffix in new[] { "weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0" })
            d[$"g.{suffix}"] = w[$"{gruPrefix}.{suffix}_reverse"];
        return d;
    }

    private float[] DecodeF0(float* posterior, int frames)
    {
        int bins = _cfg.NumBins;
        int win = _cfg.LocalAverageWindow;
        float[] f0 = new float[frames];
        for (int t = 0; t < frames; t++)
        {
            long rowOff = (long)t * bins;
            int argmax = 0;
            float peak = posterior[rowOff];
            for (int k = 1; k < bins; k++)
            {
                float v = posterior[rowOff + k];
                if (v > peak) { peak = v; argmax = k; }
            }
            if (peak < _cfg.VoicingThreshold) { f0[t] = 0f; continue; }
            int lo = Math.Max(0, argmax - win);
            int hi = Math.Min(bins - 1, argmax + win);
            float wsum = 0f, fsum = 0f;
            for (int k = lo; k <= hi; k++)
            {
                float wgt = posterior[rowOff + k];
                wsum += wgt;
                fsum += wgt * _centsMapping[k];
            }
            f0[t] = wsum > 0 ? fsum / wsum : 0f;
        }
        return f0;
    }

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}

/// <summary>BatchNorm2d folded to per-channel affine (<c>scale = γ/√(σ²+ε)</c>, <c>shift = β − μ·scale</c>),
/// applied in place over a channels-first <c>[1, C, H*W]</c> layout.</summary>
internal sealed unsafe class BatchNorm2d
{
    private readonly float[] _scale;
    private readonly float[] _shift;
    private readonly Tensor _w, _b, _rm, _rv;
    private BatchNorm2d(float[] scale, float[] shift, Tensor w, Tensor b, Tensor rm, Tensor rv)
    { _scale = scale; _shift = shift; _w = w; _b = b; _rm = rm; _rv = rv; }

    public static BatchNorm2d Load(IReadOnlyDictionary<string, Tensor> w, string prefix, float eps = 1e-5f)
    {
        Tensor gamma = WhisperOps.EnsureF32(w[$"{prefix}.weight"]);
        Tensor beta = WhisperOps.EnsureF32(w[$"{prefix}.bias"]);
        Tensor mean = WhisperOps.EnsureF32(w[$"{prefix}.running_mean"]);
        Tensor var = WhisperOps.EnsureF32(w[$"{prefix}.running_var"]);
        int c = (int)gamma.ElementCount;
        float[] scale = new float[c], shift = new float[c];
        float* g = (float*)gamma.DataPointer, b2 = (float*)beta.DataPointer;
        float* m = (float*)mean.DataPointer, v = (float*)var.DataPointer;
        for (int i = 0; i < c; i++)
        {
            scale[i] = g[i] / MathF.Sqrt(v[i] + eps);
            shift[i] = b2[i] - m[i] * scale[i];
        }
        return new BatchNorm2d(scale, shift, gamma, beta, mean, var);
    }

    public void Apply(Tensor x, int channels, int hw)
    {
        float* xp = (float*)x.DataPointer;
        for (int c = 0; c < channels; c++)
        {
            float s = _scale[c], sh = _shift[c];
            float* row = xp + (long)c * hw;
            for (int i = 0; i < hw; i++) row[i] = row[i] * s + sh;
        }
    }

    public IEnumerable<Tensor> Weights() { yield return _w; yield return _b; yield return _rm; yield return _rv; }
}

/// <summary>ConvBlockRes: <c>Conv(3×3,pad1,no-bias) → BN → ReLU → Conv → BN → ReLU</c> plus a skip
/// (identity, or a 1×1 conv when <c>in≠out</c>). Keys: <c>conv.0/3</c> (convs), <c>conv.1/4</c> (BN),
/// <c>shortcut</c> (optional 1×1).</summary>
internal sealed unsafe class ConvBlockRes
{
    private readonly int _inC, _outC;
    private Tensor _c0 = null!, _c3 = null!;
    private BatchNorm2d _bn1 = null!, _bn4 = null!;
    private Tensor? _scW, _scB;

    private ConvBlockRes(int inC, int outC) { _inC = inC; _outC = outC; }

    public static ConvBlockRes Load(IReadOnlyDictionary<string, Tensor> w, string prefix, int inC, int outC)
    {
        ConvBlockRes b = new(inC, outC)
        {
            _c0 = WhisperOps.EnsureF32(w[$"{prefix}.conv.0.weight"]),
            _bn1 = BatchNorm2d.Load(w, $"{prefix}.conv.1"),
            _c3 = WhisperOps.EnsureF32(w[$"{prefix}.conv.3.weight"]),
            _bn4 = BatchNorm2d.Load(w, $"{prefix}.conv.4"),
        };
        if (w.TryGetValue($"{prefix}.shortcut.weight", out Tensor? sc))
        {
            b._scW = WhisperOps.EnsureF32(sc);
            b._scB = w.TryGetValue($"{prefix}.shortcut.bias", out Tensor? scb) ? WhisperOps.EnsureF32(scb) : null;
        }
        return b;
    }

    public Tensor Forward(IBackend backend, Tensor x, int hh, int ww)
    {
        Tensor a = new(new TensorShape(1, _outC, hh, ww), DType.F32);
        backend.Conv2D(a, x, _c0, null, 1, 1, 1, 1);
        _bn1.Apply(a, _outC, hh * ww);
        backend.LeakyRelu(a, a, 0f);
        Tensor b = new(new TensorShape(1, _outC, hh, ww), DType.F32);
        backend.Conv2D(b, a, _c3, null, 1, 1, 1, 1);
        a.Dispose();
        _bn4.Apply(b, _outC, hh * ww);
        backend.LeakyRelu(b, b, 0f);

        // Skip path.
        Tensor skip;
        bool ownSkip;
        if (_scW is not null)
        {
            skip = new(new TensorShape(1, _outC, hh, ww), DType.F32);
            backend.Conv2D(skip, x, _scW!, _scB, 1, 1, 0, 0);
            ownSkip = true;
        }
        else { skip = x; ownSkip = false; }

        Tensor outp = new(new TensorShape(1, _outC, hh, ww), DType.F32);
        backend.Add(outp, b, skip);
        b.Dispose();
        if (ownSkip) skip.Dispose();
        return outp;
    }

    public IEnumerable<Tensor> Weights()
    {
        yield return _c0; yield return _c3;
        foreach (Tensor t in _bn1.Weights()) yield return t;
        foreach (Tensor t in _bn4.Weights()) yield return t;
        if (_scW is not null) yield return _scW;
        if (_scB is not null) yield return _scB;
    }
}

/// <summary>ResEncoderBlock: <c>n_blocks</c> ConvBlockRes then (optionally) AvgPool2d(2×2). Returns the pre-pool
/// tensor (the decoder skip) and the pooled tensor.</summary>
internal sealed unsafe class ResEncoderBlock
{
    private readonly bool _pool;
    private readonly int _outC;
    private ConvBlockRes[] _blocks = [];

    private ResEncoderBlock(int outC, bool pool) { _outC = outC; _pool = pool; }

    public static ResEncoderBlock Load(IReadOnlyDictionary<string, Tensor> w, string prefix,
        int inC, int outC, int nBlocks, bool pool)
    {
        ResEncoderBlock b = new(outC, pool) { _blocks = new ConvBlockRes[nBlocks] };
        b._blocks[0] = ConvBlockRes.Load(w, $"{prefix}.conv.0", inC, outC);
        for (int i = 1; i < nBlocks; i++)
            b._blocks[i] = ConvBlockRes.Load(w, $"{prefix}.conv.{i}", outC, outC);
        return b;
    }

    /// <summary>Returns (pre-pool, pooled). When <c>pool=false</c> both refer to the same data (pooled == clone).</summary>
    public (Tensor pre, Tensor pooled) Forward(IBackend backend, Tensor x, int hh, int ww)
    {
        Tensor cur = x;
        bool first = true;
        foreach (ConvBlockRes blk in _blocks)
        {
            Tensor next = blk.Forward(backend, cur, hh, ww);
            if (!first) cur.Dispose();
            cur = next;
            first = false;
        }
        if (!_pool) return (cur, cur);
        Tensor pooled = AvgPool2x2(cur, _outC, hh, ww);
        return (cur, pooled);
    }

    private static unsafe Tensor AvgPool2x2(Tensor x, int c, int hh, int ww)
    {
        int oh = hh / 2, ow = ww / 2;
        Tensor o = new(new TensorShape(1, c, oh, ow), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int ch = 0; ch < c; ch++)
        {
            float* src = xp + (long)ch * hh * ww;
            float* dst = op + (long)ch * oh * ow;
            for (int i = 0; i < oh; i++)
                for (int j = 0; j < ow; j++)
                {
                    int si = i * 2, sj = j * 2;
                    float s = src[(long)si * ww + sj] + src[(long)si * ww + sj + 1]
                            + src[(long)(si + 1) * ww + sj] + src[(long)(si + 1) * ww + sj + 1];
                    dst[(long)i * ow + j] = s * 0.25f;
                }
        }
        return o;
    }

    public IEnumerable<Tensor> Weights()
    {
        foreach (ConvBlockRes b in _blocks) foreach (Tensor t in b.Weights()) yield return t;
    }
}

/// <summary>ResDecoderBlock: <c>ConvTranspose2d(3×3,stride2,pad1,out_pad1,no-bias) → BN → ReLU → cat(skip) →
/// n_blocks ConvBlockRes</c>. Keys: <c>conv1.0</c> (transpose conv), <c>conv1.1</c> (BN), <c>conv2.M</c>.</summary>
internal sealed unsafe class ResDecoderBlock
{
    private readonly int _inC, _outC;
    private Tensor _up = null!;
    private BatchNorm2d _bn = null!;
    private ConvBlockRes[] _blocks = [];

    private ResDecoderBlock(int inC, int outC) { _inC = inC; _outC = outC; }

    public static ResDecoderBlock Load(IReadOnlyDictionary<string, Tensor> w, string prefix,
        int inC, int outC, int nBlocks)
    {
        ResDecoderBlock b = new(inC, outC)
        {
            _up = WhisperOps.EnsureF32(w[$"{prefix}.conv1.0.weight"]),
            _bn = BatchNorm2d.Load(w, $"{prefix}.conv1.1"),
            _blocks = new ConvBlockRes[nBlocks],
        };
        b._blocks[0] = ConvBlockRes.Load(w, $"{prefix}.conv2.0", outC * 2, outC);
        for (int i = 1; i < nBlocks; i++)
            b._blocks[i] = ConvBlockRes.Load(w, $"{prefix}.conv2.{i}", outC, outC);
        return b;
    }

    public Tensor Forward(IBackend backend, Tensor x, int hh, int ww, Tensor skip)
    {
        int oh = hh * 2, ow = ww * 2;
        // ConvTranspose2d(stride 2, pad 1, output_padding 1) -> exactly 2× input size.
        Tensor up = new(new TensorShape(1, _outC, oh, ow), DType.F32);
        backend.ConvTranspose2d(up, x, _up, null, 2, 2, 1, 1);
        _bn.Apply(up, _outC, oh * ow);
        backend.LeakyRelu(up, up, 0f);

        // Concat skip along channels -> [1, 2*outC, oh, ow].
        Tensor cat = new(new TensorShape(1, _outC * 2, oh, ow), DType.F32);
        float* up_p = (float*)up.DataPointer;
        float* sk_p = (float*)skip.DataPointer;
        float* cp = (float*)cat.DataPointer;
        long plane = (long)oh * ow;
        Buffer.MemoryCopy(up_p, cp, _outC * plane * 4, _outC * plane * 4);
        Buffer.MemoryCopy(sk_p, cp + _outC * plane, _outC * plane * 4, _outC * plane * 4);
        up.Dispose();

        Tensor cur = cat;
        foreach (ConvBlockRes blk in _blocks)
        {
            Tensor next = blk.Forward(backend, cur, oh, ow);
            cur.Dispose();
            cur = next;
        }
        return cur;
    }

    public IEnumerable<Tensor> Weights()
    {
        yield return _up;
        foreach (Tensor t in _bn.Weights()) yield return t;
        foreach (ConvBlockRes b in _blocks) foreach (Tensor t in b.Weights()) yield return t;
    }
}
