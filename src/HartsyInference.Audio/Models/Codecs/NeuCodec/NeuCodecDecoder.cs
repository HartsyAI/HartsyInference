using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.NeuCodec;

/// <summary>NeuCodec decode path (the part NeuTTS needs): FSQ index → de-quantized 8-d code → project-out
/// (8→2048) → fc_post_a (2048→1024) → Vocos transformer backbone (Conv embed + ResNet pre-net + 12 RoPE
/// transformer blocks + final LayerNorm + ResNet post-net) → iSTFT head → 24 kHz PCM. Reuses
/// <see cref="Fsq"/> (index unpack), <see cref="IStft"/> (head), <see cref="RotaryEmbedding"/> (RoPE) and the
/// shared <c>IBackend</c> conv/norm/attention ops.
///
/// <para>Structural scaffold (validation-pending): the data flow and shapes are exact; the precise checkpoint
/// key spelling, the RoPE convention (torchtune vs interleaved), and the iSTFT "same"-padding edge handling
/// are checkpoint-gated reconcile items.</para></summary>
public sealed unsafe class NeuCodecDecoder : IDisposable
{
    private readonly NeuCodecConfig _cfg;

    private Tensor? _projOutW, _projOutB;     // FSQ 8 → 2048
    private Tensor? _fcPostW, _fcPostB;       // 2048 → 1024
    private Tensor? _embedW, _embedB;         // Conv1d 1024→1024 k7
    private readonly ResnetWeights[] _prior;
    private readonly ResnetWeights[] _post;
    private readonly TxWeights[] _blocks;
    private Tensor? _finalNormW, _finalNormB;
    private Tensor? _headW, _headB;           // 1024 → n_fft+2
    private int _disposed;

    public NeuCodecConfig Config => _cfg;
    public int SampleRate => _cfg.SampleRate;
    public int FrameRate => _cfg.FrameRate;
    public int NCodebooks => 1;

    public NeuCodecDecoder(NeuCodecConfig cfg)
    {
        _cfg = cfg;
        _prior = new ResnetWeights[cfg.PriorResnetBlocks];
        _post = new ResnetWeights[cfg.PostResnetBlocks];
        _blocks = new TxWeights[cfg.Depth];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "generator")
    {
        _projOutW = WhisperOps.EnsureF32(w[$"{prefix}.quantizer.project_out.weight"]);
        _projOutB = TryGet(w, $"{prefix}.quantizer.project_out.bias");
        _fcPostW = WhisperOps.EnsureF32(w["fc_post_a.weight"]);
        _fcPostB = TryGet(w, "fc_post_a.bias");

        _embedW = ConvW(w[$"{prefix}.backbone.embed.weight"], _cfg.BackboneDim, _cfg.BackboneDim, _cfg.EmbedKernel);
        _embedB = TryGet(w, $"{prefix}.backbone.embed.bias");
        for (int i = 0; i < _prior.Length; i++) _prior[i] = LoadResnet(w, $"{prefix}.backbone.prior_net.{i}");
        for (int i = 0; i < _post.Length; i++) _post[i] = LoadResnet(w, $"{prefix}.backbone.post_net.{i}");
        for (int i = 0; i < _blocks.Length; i++) _blocks[i] = LoadTx(w, $"{prefix}.backbone.transformers.{i}");
        _finalNormW = WhisperOps.EnsureF32(w[$"{prefix}.backbone.final_layer_norm.weight"]);
        _finalNormB = WhisperOps.EnsureF32(w[$"{prefix}.backbone.final_layer_norm.bias"]);
        _headW = WhisperOps.EnsureF32(w[$"{prefix}.head.out.weight"]);
        _headB = TryGet(w, $"{prefix}.head.out.bias");
    }

    /// <summary>Decodes a single-codebook code stream <c>[F]</c> (FSQ indices) to 24 kHz mono PCM.</summary>
    public float[] Decode(IBackend backend, ReadOnlySpan<int> codes)
    {
        int f = codes.Length;
        int dim = _cfg.BackboneDim;

        // FSQ index → 8-d code vector [1, F, 8] → project_out → fc_post_a → [1, F, 1024].
        Tensor idx = new(new TensorShape(1, f), DType.I32);
        int* ip = (int*)idx.DataPointer;
        for (int i = 0; i < f; i++) ip[i] = codes[i];
        Tensor fsq = new(new TensorShape(1, f, _cfg.FsqDim), DType.F32);
        Fsq.Dequantize(fsq, idx, [.. _cfg.FsqLevels]);
        idx.Dispose();
        Tensor q = WhisperOps.ProjectLinear(backend, fsq, _projOutW!, _projOutB, 1, f, _cfg.FsqDim, _cfg.QuantizerDim);
        fsq.Dispose();
        Tensor xCl = WhisperOps.ProjectLinear(backend, q, _fcPostW!, _fcPostB, 1, f, _cfg.QuantizerDim, dim);
        q.Dispose();

        // embed: Conv1d (channels-first), then prior ResNet pre-net.
        Tensor xCf = ToChannelsFirst(backend, xCl, f, dim); xCl.Dispose();
        Tensor emb = Conv1d(backend, xCf, _embedW!, _embedB, dim, dim, f, _cfg.EmbedKernel); xCf.Dispose();
        foreach (ResnetWeights r in _prior) { Tensor n = Resnet(backend, emb, r, dim, f); emb.Dispose(); emb = n; }

        // Transformer stack (channels-last).
        Tensor t = ToChannelsLast(backend, emb, dim, f); emb.Dispose();
        (float[] cos, float[] sin) = RotaryEmbedding.GetTables(_cfg.HeadDim, _cfg.RopeTheta, Math.Max(1, f));
        foreach (TxWeights b in _blocks) { Tensor n = TxBlock(backend, t, b, f, cos, sin); t.Dispose(); t = n; }
        Tensor ln = new(t.Shape, DType.F32);
        backend.LayerNorm(ln, t, _finalNormW!, _finalNormB!, _cfg.NormEps); t.Dispose();

        // Post ResNet (channels-first).
        Tensor pCf = ToChannelsFirst(backend, ln, f, dim); ln.Dispose();
        foreach (ResnetWeights r in _post) { Tensor n = Resnet(backend, pCf, r, dim, f); pCf.Dispose(); pCf = n; }
        Tensor headIn = ToChannelsLast(backend, pCf, dim, f); pCf.Dispose();

        // Head: Linear(1024 → n_fft+2) → split mag/phase → iSTFT.
        int outDim = _cfg.NFft + 2;
        Tensor head = WhisperOps.ProjectLinear(backend, headIn, _headW!, _headB, 1, f, dim, outDim);
        headIn.Dispose();
        int half = (_cfg.NFft / 2) + 1;
        float* hp = (float*)head.DataPointer;
        float[] re = new float[(long)f * half];
        float[] im = new float[(long)f * half];
        for (int frame = 0; frame < f; frame++)
        {
            int src = frame * outDim, dst = frame * half;
            for (int k = 0; k < half; k++)
            {
                float mag = MathF.Exp(hp[src + k]);
                if (mag > 100f) mag = 100f;
                float ph = hp[src + half + k];
                re[dst + k] = mag * MathF.Cos(ph);
                im[dst + k] = mag * MathF.Sin(ph);
            }
        }
        head.Dispose();
        return IStft.Apply(re, im, f, _cfg.NFft, _cfg.HopLength);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_projOutW, _projOutB, _fcPostW, _fcPostB, _embedW, _embedB, _finalNormW, _finalNormB, _headW, _headB];
        foreach (Tensor? x in top) if (x is not null) yield return x;
        foreach (ResnetWeights r in _prior) foreach (Tensor x in r.All()) yield return x;
        foreach (ResnetWeights r in _post) foreach (Tensor x in r.All()) yield return x;
        foreach (TxWeights b in _blocks) foreach (Tensor x in b.All()) yield return x;
    }

    // ── ResNet block: GroupNorm+Silu+Conv1d ×2 + residual (channels-first [1,C,T]) ──
    private Tensor Resnet(IBackend backend, Tensor x, ResnetWeights r, int c, int t)
    {
        Tensor h1 = GroupNormSiluConv(backend, x, r.Norm1W!, r.Norm1B!, r.Conv1W!, r.Conv1B, c, t);
        Tensor h2 = GroupNormSiluConv(backend, h1, r.Norm2W!, r.Norm2B!, r.Conv2W!, r.Conv2B, c, t); h1.Dispose();
        Tensor outT = new(new TensorShape(1, c, t), DType.F32);
        backend.Add(outT, x, h2); h2.Dispose();
        return outT;
    }

    private Tensor GroupNormSiluConv(IBackend backend, Tensor x, Tensor nw, Tensor nb, Tensor cw, Tensor? cb, int c, int t)
    {
        Tensor n = new(new TensorShape(1, c, t), DType.F32);
        backend.GroupNorm(n, x, nw, nb, _cfg.GroupNormGroups, _cfg.NormEps);
        Tensor a = new(n.Shape, DType.F32);
        backend.Silu(a, n); n.Dispose();
        Tensor outT = Conv1d(backend, a, cw, cb, c, c, t, _cfg.ResnetKernel); a.Dispose();
        return outT;
    }

    // ── Transformer block: RMSNorm + RoPE MHA + RMSNorm + SiLU MLP (channels-last [1,T,dim]) ──
    private Tensor TxBlock(IBackend backend, Tensor x, TxWeights b, int t, float[] cos, float[] sin)
    {
        int dim = _cfg.BackboneDim, heads = _cfg.NumHeads, hd = _cfg.HeadDim;
        Tensor pre = new(x.Shape, DType.F32);
        backend.RmsNorm(pre, x, b.AttNormW!, _cfg.NormEps);

        Tensor qkv = WhisperOps.ProjectLinear(backend, pre, b.QkvW!, null, 1, t, dim, 3 * dim); pre.Dispose();
        Tensor qMh = new(new TensorShape(1, heads, t, hd), DType.F32);
        Tensor kMh = new(new TensorShape(1, heads, t, hd), DType.F32);
        Tensor vMh = new(new TensorShape(1, heads, t, hd), DType.F32);
        SplitQkvToHeads(qkv, qMh, kMh, vMh, t, heads, hd, dim); qkv.Dispose();
        RotaryEmbedding.ApplyInPlace(qMh, heads, t, hd, hd, 0, cos, sin);
        RotaryEmbedding.ApplyInPlace(kMh, heads, t, hd, hd, 0, cos, sin);

        Tensor attn = new(new TensorShape(1, heads, t, hd), DType.F32);
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, null, 1f / MathF.Sqrt(hd));
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();
        Tensor attnFlat = new(new TensorShape(1, t, dim), DType.F32);
        HeadsToFlat(attn, attnFlat, t, heads, hd); attn.Dispose();
        Tensor attnOut = WhisperOps.ProjectLinear(backend, attnFlat, b.ProjW!, null, 1, t, dim, dim); attnFlat.Dispose();

        Tensor afterAttn = new(x.Shape, DType.F32);
        backend.Add(afterAttn, x, attnOut); attnOut.Dispose();

        Tensor pre2 = new(x.Shape, DType.F32);
        backend.RmsNorm(pre2, afterAttn, b.FfnNormW!, _cfg.NormEps);
        Tensor h1 = WhisperOps.ProjectLinear(backend, pre2, b.Fc1W!, null, 1, t, dim, _cfg.MlpDim); pre2.Dispose();
        Tensor act = new(h1.Shape, DType.F32);
        backend.Silu(act, h1); h1.Dispose();
        Tensor h2 = WhisperOps.ProjectLinear(backend, act, b.Fc2W!, null, 1, t, _cfg.MlpDim, dim); act.Dispose();
        Tensor outT = new(x.Shape, DType.F32);
        backend.Add(outT, afterAttn, h2); afterAttn.Dispose(); h2.Dispose();
        return outT;
    }

    // ── conv / layout helpers ──
    private static Tensor Conv1d(IBackend backend, Tensor xCf, Tensor w, Tensor? b, int inC, int outC, int t, int k)
    {
        Tensor x4 = xCf.Reshape(new TensorShape(1, inC, 1, t));
        Tensor o4 = new(new TensorShape(1, outC, 1, t), DType.F32);
        backend.Conv2D(o4, x4, w, b, 1, 1, 0, k / 2);
        return o4.Reshape(new TensorShape(1, outC, t));
    }

    private static Tensor ToChannelsFirst(IBackend backend, Tensor cl, int t, int c)
    {
        Tensor cf = new(new TensorShape(1, c, t), DType.F32);
        backend.Transpose2D(cf, cl, t, c);
        return cf;
    }

    private static Tensor ToChannelsLast(IBackend backend, Tensor cf, int c, int t)
    {
        Tensor cl = new(new TensorShape(1, t, c), DType.F32);
        backend.Transpose2D(cl, cf, c, t);
        return cl;
    }

    private static void SplitQkvToHeads(Tensor qkv, Tensor q, Tensor k, Tensor v, int t, int heads, int hd, int dim)
    {
        float* src = (float*)qkv.DataPointer;
        CopyHeads(src, (float*)q.DataPointer, 0, t, heads, hd, dim);
        CopyHeads(src, (float*)k.DataPointer, dim, t, heads, hd, dim);
        CopyHeads(src, (float*)v.DataPointer, 2 * dim, t, heads, hd, dim);
    }

    private static void CopyHeads(float* src, float* dst, int colOffset, int t, int heads, int hd, int dim)
    {
        for (int s = 0; s < t; s++)
            for (int h = 0; h < heads; h++)
            {
                long srcOff = (long)s * (3 * dim) + colOffset + (long)h * hd;
                long dstOff = ((long)h * t + s) * hd;
                Buffer.MemoryCopy(src + srcOff, dst + dstOff, hd * 4, hd * 4);
            }
    }

    private static void HeadsToFlat(Tensor attn, Tensor flat, int t, int heads, int hd)
    {
        float* ip = (float*)attn.DataPointer;
        float* op = (float*)flat.DataPointer;
        int dim = heads * hd;
        for (int s = 0; s < t; s++)
            for (int h = 0; h < heads; h++)
            {
                long inOff = ((long)h * t + s) * hd;
                long outOff = (long)s * dim + (long)h * hd;
                Buffer.MemoryCopy(ip + inOff, op + outOff, hd * 4, hd * 4);
            }
    }

    private ResnetWeights LoadResnet(IReadOnlyDictionary<string, Tensor> w, string p) => new()
    {
        Norm1W = WhisperOps.EnsureF32(w[$"{p}.norm1.weight"]),
        Norm1B = WhisperOps.EnsureF32(w[$"{p}.norm1.bias"]),
        Conv1W = ConvW(w[$"{p}.conv1.weight"], _cfg.BackboneDim, _cfg.BackboneDim, _cfg.ResnetKernel),
        Conv1B = TryGet(w, $"{p}.conv1.bias"),
        Norm2W = WhisperOps.EnsureF32(w[$"{p}.norm2.weight"]),
        Norm2B = WhisperOps.EnsureF32(w[$"{p}.norm2.bias"]),
        Conv2W = ConvW(w[$"{p}.conv2.weight"], _cfg.BackboneDim, _cfg.BackboneDim, _cfg.ResnetKernel),
        Conv2B = TryGet(w, $"{p}.conv2.bias"),
    };

    private static TxWeights LoadTx(IReadOnlyDictionary<string, Tensor> w, string p) => new()
    {
        AttNormW = WhisperOps.EnsureF32(w[$"{p}.att_norm.weight"]),
        QkvW = WhisperOps.EnsureF32(w[$"{p}.att.c_attn.weight"]),
        ProjW = WhisperOps.EnsureF32(w[$"{p}.att.c_proj.weight"]),
        FfnNormW = WhisperOps.EnsureF32(w[$"{p}.ffn_norm.weight"]),
        Fc1W = WhisperOps.EnsureF32(w[$"{p}.mlp.fc1.weight"]),
        Fc2W = WhisperOps.EnsureF32(w[$"{p}.mlp.fc2.weight"]),
    };

    private static Tensor ConvW(Tensor raw, int outC, int inC, int k)
        => WhisperOps.EnsureF32(raw).Reshape(new TensorShape(outC, inC, 1, k));

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private sealed class ResnetWeights
    {
        public Tensor? Norm1W, Norm1B, Conv1W, Conv1B, Norm2W, Norm2B, Conv2W, Conv2B;
        public IEnumerable<Tensor> All()
        {
            Tensor?[] a = [Norm1W, Norm1B, Conv1W, Conv1B, Norm2W, Norm2B, Conv2W, Conv2B];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    private sealed class TxWeights
    {
        public Tensor? AttNormW, QkvW, ProjW, FfnNormW, Fc1W, Fc2W;
        public IEnumerable<Tensor> All()
        {
            Tensor?[] a = [AttNormW, QkvW, ProjW, FfnNormW, Fc1W, Fc2W];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }
}
