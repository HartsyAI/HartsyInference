using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Hubert;

/// <summary>HuBERT / Wav2Vec2-base content encoder: conv feature extractor (7 strided Conv1d, GroupNorm after
/// layer 0, GELU) → feature projection (LayerNorm + Linear 512→768) → positional conv embedding (grouped
/// Conv1d k128) → 12 post-LayerNorm transformer layers → <c>last_hidden_state [1, 768, T]</c> (channels-first,
/// the content features GPT-SoVITS / RVC consume). Reuses <see cref="IBackend"/> Conv1d / GroupNorm /
/// LayerNorm / GELU / SDPA.</summary>
public sealed unsafe class Hubert : IDisposable
{
    private readonly HubertConfig _cfg;
    private readonly Tensor?[] _convW;              // [7] each [out, in, k]
    private Tensor? _gnW, _gnB;                     // GroupNorm after conv 0
    private Tensor? _fpNormW, _fpNormB, _fpW, _fpB; // feature projection
    private Tensor? _posConvW, _posConvB;           // positional conv (grouped)
    private Tensor? _encNormW, _encNormB;           // encoder pre-stack LayerNorm
    private readonly Layer[] _layers;
    private int _disposed;

    public HubertConfig Config => _cfg;

    public Hubert(HubertConfig cfg)
    {
        _cfg = cfg;
        _convW = new Tensor?[cfg.ConvKernels.Count];
        _layers = new Layer[cfg.NumLayers];
        for (int i = 0; i < _layers.Length; i++) _layers[i] = new Layer(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        for (int i = 0; i < _convW.Length; i++)
            _convW[i] = WhisperOps.EnsureF32(w[$"{prefix}feature_extractor.conv_layers.{i}.conv.weight"]);
        _gnW = WhisperOps.EnsureF32(w[$"{prefix}feature_extractor.conv_layers.0.layer_norm.weight"]);
        _gnB = WhisperOps.EnsureF32(w[$"{prefix}feature_extractor.conv_layers.0.layer_norm.bias"]);
        _fpNormW = WhisperOps.EnsureF32(w[$"{prefix}feature_projection.layer_norm.weight"]);
        _fpNormB = WhisperOps.EnsureF32(w[$"{prefix}feature_projection.layer_norm.bias"]);
        _fpW = WhisperOps.EnsureF32(w[$"{prefix}feature_projection.projection.weight"]);
        _fpB = WhisperOps.EnsureF32(w[$"{prefix}feature_projection.projection.bias"]);
        // pos_conv is weight-normed in the real wav2vec2/HuBERT checkpoints (dim=2 → one magnitude per kernel
        // position). Compose weight_g[1,1,k]·weight_v[out,in,k]/‖weight_v[:,:,k]‖ when the raw conv.weight is absent.
        _posConvW = w.TryGetValue($"{prefix}encoder.pos_conv_embed.conv.weight", out Tensor? pcw)
            ? WhisperOps.EnsureF32(pcw)
            : ComposePosConvWeightNorm(
                WhisperOps.EnsureF32(w[$"{prefix}encoder.pos_conv_embed.conv.weight_g"]),
                WhisperOps.EnsureF32(w[$"{prefix}encoder.pos_conv_embed.conv.weight_v"]));
        _posConvB = w.TryGetValue($"{prefix}encoder.pos_conv_embed.conv.bias", out Tensor? pb) ? WhisperOps.EnsureF32(pb) : null;
        _encNormW = WhisperOps.EnsureF32(w[$"{prefix}encoder.layer_norm.weight"]);
        _encNormB = WhisperOps.EnsureF32(w[$"{prefix}encoder.layer_norm.bias"]);
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}encoder.layers.{i}");
    }

    /// <summary>Exact (erf-based) GELU in place: <c>0.5·x·(1 + erf(x/√2))</c> (Abramowitz-Stegun 7.1.26,
    /// max abs error ≈ 1.5e-7). HuBERT/wav2vec2 use the exact GELU, not the tanh approximation.</summary>
    internal static void ExactGelu(float* p, long n)
    {
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, pp = 0.3275911f, invSqrt2 = 0.70710678f;
        for (long i = 0; i < n; i++)
        {
            float x = p[i], z = x * invSqrt2;
            float sign = z < 0 ? -1f : 1f, az = MathF.Abs(z);
            float tt = 1f / (1f + pp * az);
            float poly = ((((a5 * tt + a4) * tt + a3) * tt + a2) * tt + a1) * tt;
            float erf = sign * (1f - poly * MathF.Exp(-az * az));
            p[i] = 0.5f * x * (1f + erf);
        }
    }

    /// <summary>Composes a <c>weight_norm(dim=2)</c> grouped Conv1d weight: for each kernel position k,
    /// <c>W[:,:,k] = V[:,:,k] · g[0,0,k] / ‖V[:,:,k]‖_F</c>. <paramref name="g"/> is <c>[1,1,K]</c>,
    /// <paramref name="v"/> is <c>[outC, inC/groups, K]</c>.</summary>
    private static Tensor ComposePosConvWeightNorm(Tensor g, Tensor v)
    {
        int outC = (int)v.Shape[0], inC = (int)v.Shape[1], k = (int)v.Shape[2];
        Tensor outT = new(v.Shape, DType.F32);
        float* gp = (float*)g.DataPointer, vp = (float*)v.DataPointer, op = (float*)outT.DataPointer;
        for (int kk = 0; kk < k; kk++)
        {
            double sumSq = 0;
            for (int o = 0; o < outC; o++)
                for (int i = 0; i < inC; i++)
                {
                    float val = vp[((long)o * inC + i) * k + kk];
                    sumSq += (double)val * val;
                }
            float scale = (float)(gp[kk] / (Math.Sqrt(sumSq) + 1e-12));
            for (int o = 0; o < outC; o++)
                for (int i = 0; i < inC; i++)
                {
                    long idx = ((long)o * inC + i) * k + kk;
                    op[idx] = vp[idx] * scale;
                }
        }
        return outT;
    }

    /// <summary>Encodes 16 kHz mono PCM <c>[1, 1, T_pcm]</c> → the MEAN over all 13 hidden states (the encoder
    /// input after pos-conv + LayerNorm, plus each of the 12 layer outputs), channels-first <c>[1, hidden, T]</c>.
    /// This is <c>SoundStream.get_regress_target</c> — x-codec averages every state, NOT the layer-9 feature the
    /// usual RepCodec/HuBERT-tokenizer convention uses. The caller supplies the 160-sample zero pad per side.</summary>
    public Tensor ForwardMeanHiddenStates(IBackend backend, Tensor pcm, int tPcm)
        => Forward(backend, pcm, tPcm, meanHiddenStates: true);

    /// <summary>Encodes 16 kHz mono PCM <c>[1, 1, T_pcm]</c> → content features <c>[1, hidden, T]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor pcm, int tPcm) => Forward(backend, pcm, tPcm, meanHiddenStates: false);

    private Tensor Forward(IBackend backend, Tensor pcm, int tPcm, bool meanHiddenStates)
    {
        // Conv feature extractor → [1, 512, T'].
        Tensor x = ConvExtractor(backend, pcm, tPcm, out int tFrames);

        // Channels-last [1, T', 512] for feature projection.
        Tensor xl = new(new TensorShape(1, tFrames, _cfg.ConvDim), DType.F32);
        backend.Transpose2D(xl, x, _cfg.ConvDim, tFrames); x.Dispose();
        Tensor normed = new(xl.Shape, DType.F32);
        backend.LayerNorm(normed, xl, _fpNormW!, _fpNormB!, _cfg.NormEps); xl.Dispose();
        Tensor h = WhisperOps.ProjectLinear(backend, normed, _fpW!, _fpB, 1, tFrames, _cfg.ConvDim, _cfg.Hidden);
        normed.Dispose();

        // Positional conv embed (channels-first, grouped) + GELU, added to h.
        Tensor hcf = new(new TensorShape(1, _cfg.Hidden, tFrames), DType.F32);
        backend.Transpose2D(hcf, h, tFrames, _cfg.Hidden);
        Tensor pos4 = hcf.Reshape(new TensorShape(1, _cfg.Hidden, 1, tFrames));
        int pad = _cfg.PosConvKernel / 2;
        Tensor posOut = new(new TensorShape(1, _cfg.Hidden, tFrames + 1), DType.F32);   // "same"+1 (k even)
        backend.Conv1d(posOut, hcf, _posConvW!, _posConvB, 1, pad, pad, 1, _cfg.PosConvGroups);
        hcf.Dispose();
        // Drop the last element (wav2vec2 num_pad_remove=1 for even kernel) + exact GELU, add to h.
        ExactGelu((float*)posOut.DataPointer, posOut.ElementCount);
        float* pp = (float*)posOut.DataPointer;
        float* hp = (float*)h.DataPointer;
        for (int c = 0; c < _cfg.Hidden; c++)
            for (int j = 0; j < tFrames; j++)
                hp[(long)j * _cfg.Hidden + c] += pp[(long)c * (tFrames + 1) + j];
        posOut.Dispose();

        // Encoder pre-stack LayerNorm (post-norm variant applies it here), then 12 layers.
        Tensor enc = new(h.Shape, DType.F32);
        backend.LayerNorm(enc, h, _encNormW!, _encNormB!, _cfg.NormEps); h.Dispose();

        Tensor? acc = null;
        if (meanHiddenStates)
        {
            acc = new Tensor(enc.Shape, DType.F32);
            long n = enc.ElementCount;
            Buffer.MemoryCopy((void*)enc.DataPointer, (void*)acc.DataPointer, n * sizeof(float), n * sizeof(float));
        }
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, enc, tFrames);
            enc.Dispose(); enc = next;
            if (acc is null) continue;
            float* ap = (float*)acc.DataPointer;
            float* np = (float*)enc.DataPointer;
            for (long j = 0; j < acc.ElementCount; j++) ap[j] += np[j];
        }
        if (acc is not null)
        {
            float scale = 1f / (_layers.Length + 1);
            float* ap = (float*)acc.DataPointer;
            for (long j = 0; j < acc.ElementCount; j++) ap[j] *= scale;
            enc.Dispose(); enc = acc;
        }

        // Return channels-first [1, hidden, T] (get_content transpose).
        Tensor outCf = new(new TensorShape(1, _cfg.Hidden, tFrames), DType.F32);
        backend.Transpose2D(outCf, enc, tFrames, _cfg.Hidden); enc.Dispose();
        return outCf;
    }

    private Tensor ConvExtractor(IBackend backend, Tensor pcm, int tPcm, out int tFrames)
    {
        Tensor cur = pcm;   // [1, 1, T_pcm]
        int inCh = 1, curT = tPcm;
        bool owns = false;
        for (int i = 0; i < _convW.Length; i++)
        {
            int k = _cfg.ConvKernels[i], s = _cfg.ConvStrides[i], outCh = _cfg.ConvDim;
            int outT = (curT - k) / s + 1;
            Tensor o = new(new TensorShape(1, outCh, outT), DType.F32);
            backend.Conv1d(o, cur, _convW[i]!, null, s, 0, 0, 1, 1);
            if (owns) cur.Dispose();
            cur = o; inCh = outCh; curT = outT; owns = true;

            if (i == 0)   // GroupNorm after conv 0 (feat_extract_norm="group")
            {
                // The CPU GroupNorm expects rank-4 [N,C,H,W]; present cur [1,C,T] as [1,C,T,1] so each channel
                // (groups == channels) is normalized over time. (A rank-3 input gives W=0 → broken norm.)
                Tensor cur4 = cur.Reshape(new TensorShape(1, outCh, outT, 1));
                Tensor gn = new(new TensorShape(1, outCh, outT, 1), DType.F32);
                backend.GroupNorm(gn, cur4, _gnW!, _gnB!, outCh, _cfg.NormEps);
                cur.Dispose(); cur = gn.Reshape(new TensorShape(1, outCh, outT));
            }
            // GELU activation after each conv (exact erf — HuBERT uses the exact GELU, not the tanh approx).
            ExactGelu((float*)cur.DataPointer, cur.ElementCount);
        }
        tFrames = curT;
        return cur;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in _convW) if (t is not null) yield return t;
        Tensor?[] own = [_gnW, _gnB, _fpNormW, _fpNormB, _fpW, _fpB, _posConvW, _posConvB, _encNormW, _encNormB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (Layer l in _layers) foreach (Tensor t in l.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    /// <summary>One post-LayerNorm transformer layer: attn → +res → LN → FFN(GELU) → +res → LN.</summary>
    private sealed class Layer
    {
        private readonly HubertConfig _cfg;
        private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _ln1W, _ln1B, _ff1W, _ff1B, _ff2W, _ff2B, _ln2W, _ln2B;

        public Layer(HubertConfig cfg) => _cfg = cfg;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _qW = WhisperOps.EnsureF32(w[$"{p}.attention.q_proj.weight"]); _qB = WhisperOps.EnsureF32(w[$"{p}.attention.q_proj.bias"]);
            _kW = WhisperOps.EnsureF32(w[$"{p}.attention.k_proj.weight"]); _kB = WhisperOps.EnsureF32(w[$"{p}.attention.k_proj.bias"]);
            _vW = WhisperOps.EnsureF32(w[$"{p}.attention.v_proj.weight"]); _vB = WhisperOps.EnsureF32(w[$"{p}.attention.v_proj.bias"]);
            _oW = WhisperOps.EnsureF32(w[$"{p}.attention.out_proj.weight"]); _oB = WhisperOps.EnsureF32(w[$"{p}.attention.out_proj.bias"]);
            _ln1W = WhisperOps.EnsureF32(w[$"{p}.layer_norm.weight"]); _ln1B = WhisperOps.EnsureF32(w[$"{p}.layer_norm.bias"]);
            _ff1W = WhisperOps.EnsureF32(w[$"{p}.feed_forward.intermediate_dense.weight"]); _ff1B = WhisperOps.EnsureF32(w[$"{p}.feed_forward.intermediate_dense.bias"]);
            _ff2W = WhisperOps.EnsureF32(w[$"{p}.feed_forward.output_dense.weight"]); _ff2B = WhisperOps.EnsureF32(w[$"{p}.feed_forward.output_dense.bias"]);
            _ln2W = WhisperOps.EnsureF32(w[$"{p}.final_layer_norm.weight"]); _ln2B = WhisperOps.EnsureF32(w[$"{p}.final_layer_norm.bias"]);
        }

        public Tensor Forward(IBackend backend, Tensor x, int t)
        {
            int h = _cfg.Hidden, heads = _cfg.NumHeads, hd = _cfg.HeadDim;
            Tensor q = WhisperOps.ProjectLinear(backend, x, _qW!, _qB, 1, t, h, h);
            Tensor k = WhisperOps.ProjectLinear(backend, x, _kW!, _kB, 1, t, h, h);
            Tensor v = WhisperOps.ProjectLinear(backend, x, _vW!, _vB, 1, t, h, h);
            Tensor qM = new(new TensorShape(1, heads, t, hd), DType.F32);
            Tensor kM = new(new TensorShape(1, heads, t, hd), DType.F32);
            Tensor vM = new(new TensorShape(1, heads, t, hd), DType.F32);
            Dia.DiaHeads.FlatToHeads(qM, q, t, heads, hd); q.Dispose();
            Dia.DiaHeads.FlatToHeads(kM, k, t, heads, hd); k.Dispose();
            Dia.DiaHeads.FlatToHeads(vM, v, t, heads, hd); v.Dispose();
            Tensor attn = new(new TensorShape(1, heads, t, hd), DType.F32);
            backend.ScaledDotProductAttention(attn, qM, kM, vM, null, 1f / MathF.Sqrt(hd));
            qM.Dispose(); kM.Dispose(); vM.Dispose();
            Tensor flat = new(new TensorShape(1, t, h), DType.F32);
            Dia.DiaHeads.HeadsToFlat(flat, attn, t, heads, hd); attn.Dispose();
            Tensor attnOut = WhisperOps.ProjectLinear(backend, flat, _oW!, _oB, 1, t, h, h); flat.Dispose();

            Tensor afterAttn = new(x.Shape, DType.F32);
            backend.Add(afterAttn, x, attnOut); attnOut.Dispose();
            Tensor n1 = new(x.Shape, DType.F32);
            backend.LayerNorm(n1, afterAttn, _ln1W!, _ln1B!, _cfg.NormEps); afterAttn.Dispose();

            Tensor f1 = WhisperOps.ProjectLinear(backend, n1, _ff1W!, _ff1B, 1, t, h, _cfg.FfnDim);
            ExactGelu((float*)f1.DataPointer, f1.ElementCount);   // exact erf GELU (HuBERT hidden_act="gelu")
            Tensor f2 = WhisperOps.ProjectLinear(backend, f1, _ff2W!, _ff2B, 1, t, _cfg.FfnDim, h); f1.Dispose();
            Tensor afterFf = new(x.Shape, DType.F32);
            backend.Add(afterFf, n1, f2); n1.Dispose(); f2.Dispose();
            Tensor outT = new(x.Shape, DType.F32);
            backend.LayerNorm(outT, afterFf, _ln2W!, _ln2B!, _cfg.NormEps); afterFf.Dispose();
            return outT;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all = [_qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _ln1W, _ln1B, _ff1W, _ff1B, _ff2W, _ff2B, _ln2W, _ln2B];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }
    }
}
