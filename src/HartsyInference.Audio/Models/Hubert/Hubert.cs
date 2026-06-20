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
        _posConvW = WhisperOps.EnsureF32(w[$"{prefix}encoder.pos_conv_embed.conv.weight"]);
        _posConvB = w.TryGetValue($"{prefix}encoder.pos_conv_embed.conv.bias", out Tensor? pb) ? WhisperOps.EnsureF32(pb) : null;
        _encNormW = WhisperOps.EnsureF32(w[$"{prefix}encoder.layer_norm.weight"]);
        _encNormB = WhisperOps.EnsureF32(w[$"{prefix}encoder.layer_norm.bias"]);
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}encoder.layers.{i}");
    }

    /// <summary>Encodes 16 kHz mono PCM <c>[1, 1, T_pcm]</c> → content features <c>[1, hidden, T]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor pcm, int tPcm)
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
        // Drop the last element (wav2vec2 num_pad_remove=1 for even kernel) + GELU, add to h.
        float* pp = (float*)posOut.DataPointer;
        float* hp = (float*)h.DataPointer;
        for (int c = 0; c < _cfg.Hidden; c++)
            for (int j = 0; j < tFrames; j++)
            {
                float v = pp[(long)c * (tFrames + 1) + j];
                float gelu = 0.5f * v * (1f + MathF.Tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
                hp[(long)j * _cfg.Hidden + c] += gelu;
            }
        posOut.Dispose();

        // Encoder pre-stack LayerNorm (post-norm variant applies it here), then 12 layers.
        Tensor enc = new(h.Shape, DType.F32);
        backend.LayerNorm(enc, h, _encNormW!, _encNormB!, _cfg.NormEps); h.Dispose();
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, enc, tFrames);
            enc.Dispose(); enc = next;
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
                Tensor gn = new(cur.Shape, DType.F32);
                backend.GroupNorm(gn, cur, _gnW!, _gnB!, outCh, _cfg.NormEps);   // groups == channels
                cur.Dispose(); cur = gn;
            }
            // GELU activation after each conv.
            float* cp = (float*)cur.DataPointer;
            for (long n = 0; n < cur.ElementCount; n++)
            {
                float v = cp[n];
                cp[n] = 0.5f * v * (1f + MathF.Tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
            }
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
            Tensor act = new(f1.Shape, DType.F32);
            backend.Gelu(act, f1); f1.Dispose();
            Tensor f2 = WhisperOps.ProjectLinear(backend, act, _ff2W!, _ff2B, 1, t, _cfg.FfnDim, h); act.Dispose();
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
