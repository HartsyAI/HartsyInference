using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Wav2Vec2 audio feature extractor (HF <c>Wav2Vec2Model</c>, post-norm encoder), the frozen front-end for
/// Wan2.2-S2V. Raw 16 kHz waveform → 7-layer strided Conv1d feature encoder → feature projection → grouped positional
/// conv → N transformer layers (bidirectional MHA + GELU FFN). <see cref="EncodeAllLayers"/> returns every hidden
/// state stacked <c>[T, numHiddenStates, hidden]</c> for the S2V audio encoder to weight + project. B=1.</summary>
public sealed unsafe class Wav2Vec2Encoder
{
    private readonly Wav2Vec2EncoderConfig _config;

    private Tensor?[] _convW = [], _convB = [];
    private Tensor? _conv0NormW, _conv0NormB;           // group norm on conv layer 0 (group mode)
    private Tensor? _fpNormW, _fpNormB, _fpProjW, _fpProjB;   // feature projection
    private Tensor? _posConvW, _posConvB;               // positional conv embedding (grouped)
    private Tensor? _encNormW, _encNormB;               // encoder layer_norm (pre-layers, non-stable)
    private readonly Layer[] _layers;

    private sealed class Layer
    {
        public Tensor? QW, QB, KW, KB, VW, VB, OW, OB;
        public Tensor? AttnNormW, AttnNormB, FfInW, FfInB, FfOutW, FfOutB, FinalNormW, FinalNormB;
    }

    public Wav2Vec2Encoder(Wav2Vec2EncoderConfig config)
    {
        _config = config;
        _layers = new Layer[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++) _layers[i] = new Layer();
    }

    public Wav2Vec2EncoderConfig Config => _config;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        int nConv = _config.ConvDims.Length;
        _convW = new Tensor?[nConv]; _convB = new Tensor?[nConv];
        for (int i = 0; i < nConv; i++)
        {
            _convW[i] = LoadF32(w, $"feature_extractor.conv_layers.{i}.conv.weight");
            w.TryGetValue($"feature_extractor.conv_layers.{i}.conv.bias", out _convB[i]);
        }
        if (_config.GroupNormFirstConvOnly)
        {
            _conv0NormW = LoadF32(w, "feature_extractor.conv_layers.0.layer_norm.weight");
            _conv0NormB = LoadF32(w, "feature_extractor.conv_layers.0.layer_norm.bias");
        }
        _fpNormW = LoadF32(w, "feature_projection.layer_norm.weight"); _fpNormB = LoadF32(w, "feature_projection.layer_norm.bias");
        _fpProjW = w["feature_projection.projection.weight"]; w.TryGetValue("feature_projection.projection.bias", out _fpProjB);
        _posConvW = LoadPosConv(w); w.TryGetValue("encoder.pos_conv_embed.conv.bias", out _posConvB);
        _encNormW = LoadF32(w, "encoder.layer_norm.weight"); _encNormB = LoadF32(w, "encoder.layer_norm.bias");
        for (int i = 0; i < _layers.Length; i++)
        {
            string p = $"encoder.layers.{i}";
            Layer l = _layers[i];
            l.QW = w[$"{p}.attention.q_proj.weight"]; w.TryGetValue($"{p}.attention.q_proj.bias", out l.QB);
            l.KW = w[$"{p}.attention.k_proj.weight"]; w.TryGetValue($"{p}.attention.k_proj.bias", out l.KB);
            l.VW = w[$"{p}.attention.v_proj.weight"]; w.TryGetValue($"{p}.attention.v_proj.bias", out l.VB);
            l.OW = w[$"{p}.attention.out_proj.weight"]; w.TryGetValue($"{p}.attention.out_proj.bias", out l.OB);
            l.AttnNormW = LoadF32(w, $"{p}.layer_norm.weight"); l.AttnNormB = LoadF32(w, $"{p}.layer_norm.bias");
            l.FfInW = w[$"{p}.feed_forward.intermediate_dense.weight"]; w.TryGetValue($"{p}.feed_forward.intermediate_dense.bias", out l.FfInB);
            l.FfOutW = w[$"{p}.feed_forward.output_dense.weight"]; w.TryGetValue($"{p}.feed_forward.output_dense.bias", out l.FfOutB);
            l.FinalNormW = LoadF32(w, $"{p}.final_layer_norm.weight"); l.FinalNormB = LoadF32(w, $"{p}.final_layer_norm.bias");
        }
    }

    /// <summary>Merged or weight-normed positional conv weight (<c>encoder.pos_conv_embed.conv.weight</c>, or the
    /// weight_norm pair <c>parametrizations.weight.original0/1</c> = g·v/‖v‖).</summary>
    private Tensor LoadPosConv(IReadOnlyDictionary<string, Tensor> w)
    {
        if (w.TryGetValue("encoder.pos_conv_embed.conv.weight", out Tensor? merged)) return LoadF32In(merged);
        Tensor g = LoadF32(w, "encoder.pos_conv_embed.conv.parametrizations.weight.original0");
        Tensor v = LoadF32(w, "encoder.pos_conv_embed.conv.parametrizations.weight.original1");
        // weight = g * v / ||v|| over the (in,k) dims per output channel (HF weight_norm dim=2 → norm over dims 0,1).
        int outC = (int)v.Shape[0], inC = (int)v.Shape[1], k = (int)v.Shape[2];
        Tensor o = new Tensor(v.Shape, DType.F32);
        float* vp = (float*)v.DataPointer, gp = (float*)g.DataPointer, op = (float*)o.DataPointer;
        // HF weight_norm dim=2: norm computed per kernel position across (out,in). Match by normalizing each [:, :, ki].
        for (int ki = 0; ki < k; ki++)
        {
            double sumSq = 0;
            for (int oc = 0; oc < outC; oc++) for (int ic = 0; ic < inC; ic++) { float val = vp[((long)oc * inC + ic) * k + ki]; sumSq += (double)val * val; }
            float norm = (float)Math.Sqrt(sumSq) + 1e-12f;
            float gk = gp[ki];
            for (int oc = 0; oc < outC; oc++) for (int ic = 0; ic < inC; ic++) { long idx = ((long)oc * inC + ic) * k + ki; op[idx] = vp[idx] * gk / norm; }
        }
        return o;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in _convW) if (t is not null) yield return t;
        foreach (Tensor? t in _convB) if (t is not null) yield return t;
        foreach (Tensor? t in new[] { _conv0NormW, _conv0NormB, _fpNormW, _fpNormB, _fpProjW, _fpProjB, _posConvW, _posConvB, _encNormW, _encNormB })
            if (t is not null) yield return t;
        foreach (Layer l in _layers)
            foreach (Tensor? t in new[] { l.QW, l.QB, l.KW, l.KB, l.VW, l.VB, l.OW, l.OB, l.AttnNormW, l.AttnNormB, l.FfInW, l.FfInB, l.FfOutW, l.FfOutB, l.FinalNormW, l.FinalNormB })
                if (t is not null) yield return t;
    }

    /// <summary>Encodes a mono 16 kHz waveform <c>[samples]</c> → all hidden states stacked <c>[T, numHiddenStates, hidden]</c>.</summary>
    public Tensor EncodeAllLayers(IBackend backend, ReadOnlySpan<float> waveform)
    {
        int samples = waveform.Length;
        Tensor feat = new Tensor(new TensorShape(1, 1, samples), DType.F32);
        fixed (float* wp = waveform) Buffer.MemoryCopy(wp, (float*)feat.DataPointer, (long)samples * 4, (long)samples * 4);

        // 1. Conv feature encoder.
        for (int i = 0; i < _convW.Length; i++)
        {
            int inC = (int)feat.Shape[1], tin = (int)feat.Shape[2];
            int outC = _config.ConvDims[i], kernel = _config.ConvKernels[i], stride = _config.ConvStrides[i];
            int tout = (tin - kernel) / stride + 1;
            Tensor conv = new Tensor(new TensorShape(1, outC, tout), DType.F32);
            backend.Conv1d(conv, feat, _convW[i]!, _convB[i], stride, 0, 0, 1, 1);
            feat.Dispose();
            if (i == 0 && _config.GroupNormFirstConvOnly)
            {
                Tensor gn = new Tensor(conv.Shape, DType.F32);
                backend.GroupNorm(gn, conv, _conv0NormW!, _conv0NormB!, outC, _config.Eps);   // groups = channels
                conv.Dispose();
                conv = gn;
            }
            backend.Gelu(conv, conv);
            feat = conv;
        }

        int t = (int)feat.Shape[2], cdim = (int)feat.Shape[1];
        // 2. [1,512,T] → [T,512], feature projection LN + Linear → [T,hidden].
        Tensor seq = ChannelsToTokens(feat, cdim, t); feat.Dispose();
        Tensor normed = new Tensor(seq.Shape, DType.F32); backend.LayerNorm(normed, seq, _fpNormW!, _fpNormB!, _config.Eps); seq.Dispose();
        Tensor hidden = new Tensor(new TensorShape(t, _config.Hidden), DType.F32); backend.Linear(hidden, normed, _fpProjW!, _fpProjB); normed.Dispose();

        // 3. Positional conv (grouped, "same" pad with even-kernel right trim) + GELU, added to hidden.
        AddPositionalConv(backend, hidden, t);

        // 4. encoder.layer_norm (non-stable: before the layers).
        Tensor h = new Tensor(hidden.Shape, DType.F32); backend.LayerNorm(h, hidden, _encNormW!, _encNormB!, _config.Eps); hidden.Dispose();

        // 5. Collect hidden states: hs[0] = h, then after each layer.
        int numStates = _config.NumHiddenStates;
        Tensor stacked = new Tensor(new TensorShape(t, numStates, _config.Hidden), DType.F32);
        WriteState(stacked, h, 0, t, numStates);
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = EncoderLayer(backend, h, _layers[i], t);
            h.Dispose();
            h = next;
            WriteState(stacked, h, i + 1, t, numStates);
        }
        h.Dispose();
        return stacked;
    }

    private void AddPositionalConv(IBackend backend, Tensor hidden, int t)
    {
        int dim = _config.Hidden, k = _config.PosConvKernel, groups = _config.PosConvGroups;
        Tensor chFirst = TokensToChannels(hidden, dim, t);                  // [1, dim, T]
        int pad = k / 2;
        int tout = t + 2 * pad - k + 1;                                     // = t+1 for even k
        Tensor pos = new Tensor(new TensorShape(1, dim, tout), DType.F32);
        backend.Conv1d(pos, chFirst, _posConvW!, _posConvB, 1, pad, pad, 1, groups);
        chFirst.Dispose();
        // "same" padding: drop the last frame when the kernel is even.
        int keep = t;
        backend.Gelu(pos, pos);
        float* pp = (float*)pos.DataPointer, hp = (float*)hidden.DataPointer;
        for (int ti = 0; ti < keep; ti++)
            for (int d = 0; d < dim; d++) hp[(long)ti * dim + d] += pp[(long)d * tout + ti];   // pos is [dim,tout] channels-first
        pos.Dispose();
    }

    private Tensor EncoderLayer(IBackend backend, Tensor h, Layer l, int t)
    {
        int dim = _config.Hidden;
        Tensor attn = Attention(backend, h, l, t);
        Tensor h1 = AddRows(h, attn, t, dim); attn.Dispose();
        Tensor n1 = new Tensor(h1.Shape, DType.F32); backend.LayerNorm(n1, h1, l.AttnNormW!, l.AttnNormB!, _config.Eps); h1.Dispose();

        int inter = (int)l.FfInW!.Shape[0];
        Tensor ff1 = new Tensor(new TensorShape(t, inter), DType.F32); backend.Linear(ff1, n1, l.FfInW!, l.FfInB);
        backend.Gelu(ff1, ff1);
        Tensor ff2 = new Tensor(new TensorShape(t, dim), DType.F32); backend.Linear(ff2, ff1, l.FfOutW!, l.FfOutB); ff1.Dispose();
        Tensor h2 = AddRows(n1, ff2, t, dim); n1.Dispose(); ff2.Dispose();
        Tensor outT = new Tensor(h2.Shape, DType.F32); backend.LayerNorm(outT, h2, l.FinalNormW!, l.FinalNormB!, _config.Eps); h2.Dispose();
        return outT;
    }

    private Tensor Attention(IBackend backend, Tensor h, Layer l, int t)
    {
        int dim = _config.Hidden, heads = _config.NumHeads, hd = dim / heads;
        Tensor q = new Tensor(new TensorShape(t, dim), DType.F32); backend.Linear(q, h, l.QW!, l.QB);
        Tensor k = new Tensor(new TensorShape(t, dim), DType.F32); backend.Linear(k, h, l.KW!, l.KB);
        Tensor v = new Tensor(new TensorShape(t, dim), DType.F32); backend.Linear(v, h, l.VW!, l.VB);
        Tensor qMh = ToBhsd(q, t, heads, hd); q.Dispose();
        Tensor kMh = ToBhsd(k, t, heads, hd); k.Dispose();
        Tensor vMh = ToBhsd(v, t, heads, hd); v.Dispose();
        Tensor attn = new Tensor(new TensorShape(1, heads, t, hd), DType.F32);
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, null, 1f / MathF.Sqrt(hd));
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();
        Tensor flat = FromBhsd(attn, t, heads, hd); attn.Dispose();
        Tensor o = new Tensor(new TensorShape(t, dim), DType.F32); backend.Linear(o, flat, l.OW!, l.OB); flat.Dispose();
        return o;
    }

    private static Tensor ToBhsd(Tensor x, int t, int heads, int hd)
    {
        Tensor o = new Tensor(new TensorShape(1, heads, t, hd), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int ti = 0; ti < t; ti++) for (int h = 0; h < heads; h++)
            Buffer.MemoryCopy(xp + (long)ti * heads * hd + (long)h * hd, op + ((long)h * t + ti) * hd, (long)hd * 4, (long)hd * 4);
        return o;
    }

    private static Tensor FromBhsd(Tensor x, int t, int heads, int hd)
    {
        Tensor o = new Tensor(new TensorShape(t, heads * hd), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int h = 0; h < heads; h++) for (int ti = 0; ti < t; ti++)
            Buffer.MemoryCopy(xp + ((long)h * t + ti) * hd, op + (long)ti * heads * hd + (long)h * hd, (long)hd * 4, (long)hd * 4);
        return o;
    }

    private static Tensor ChannelsToTokens(Tensor x, int c, int t)
    {
        Tensor o = new Tensor(new TensorShape(t, c), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++) for (int ti = 0; ti < t; ti++) op[(long)ti * c + ci] = xp[(long)ci * t + ti];
        return o;
    }

    private static Tensor TokensToChannels(Tensor x, int c, int t)
    {
        Tensor o = new Tensor(new TensorShape(1, c, t), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int ti = 0; ti < t; ti++) for (int ci = 0; ci < c; ci++) op[(long)ci * t + ti] = xp[(long)ti * c + ci];
        return o;
    }

    private static Tensor AddRows(Tensor a, Tensor b, int t, int dim)
    {
        Tensor o = new Tensor(new TensorShape(t, dim), DType.F32);
        long n = (long)t * dim;
        float* ap = (float*)a.DataPointer, bp = (float*)b.DataPointer, op = (float*)o.DataPointer;
        for (long i = 0; i < n; i++) op[i] = ap[i] + bp[i];
        return o;
    }

    private static void WriteState(Tensor stacked, Tensor h, int stateIdx, int t, int numStates)
    {
        int dim = (int)h.Shape[1];
        float* sp = (float*)stacked.DataPointer, hp = (float*)h.DataPointer;
        for (int ti = 0; ti < t; ti++)
            Buffer.MemoryCopy(hp + (long)ti * dim, sp + ((long)ti * numStates + stateIdx) * dim, (long)dim * 4, (long)dim * 4);
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
    private static Tensor LoadF32In(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);
}
