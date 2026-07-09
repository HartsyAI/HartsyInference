using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.NeuCodec;

/// <summary>NeuCodec decode path (the part NeuTTS needs), faithful to the on-disk
/// <c>neuphonic/neucodec</c> checkpoint (HF <c>NeuCodecModel</c> layout, verified against
/// transformers PR #47143 <c>modeling_neucodec.py</c>). Flow:
/// FSQ index → codebook de-quant (8-d in [-1,1]) → <c>quantizer.project_out</c> (8→2048) →
/// <c>acoustic_decoder.fc</c> (2048→1024) → Conv1d embed (k7) → 2× ResNet prior-net →
/// 12 transformer layers (RMSNorm + full bidirectional MHA + SiLU MLP) → 2× ResNet post-net →
/// final LayerNorm → ISTFT head (Linear 1024→n_fft+2, "same"-padding iSTFT) → 24 kHz PCM.
///
/// <para><b>Checkpoint key layout</b> (no <c>generator.</c> prefix): <c>quantizer.project_out.*</c>,
/// <c>acoustic_decoder.{fc,embed,prior_net.i,layers.i,post_net.i,norm,head.linear}</c>.</para>
///
/// <para><b>RoPE is intentionally skipped.</b> The reference applies rotary embeddings with
/// <c>position_ids = arange(num_heads)</c> (a per-head-constant angle, identical for q and k and
/// uniform across time). Because that rotation is orthogonal and identical on q and k, it cancels
/// in every q·kᵀ score — RoPE is a mathematical no-op here, so we omit it for bit-identical output.
/// (This mirrors the original torchtune misuse the HF port faithfully reproduces.)</para></summary>
public sealed unsafe class NeuCodecDecoder : IDisposable
{
    private readonly NeuCodecConfig _cfg;

    private Tensor? _projOutW, _projOutB;     // quantizer.project_out : FSQ 8 → 2048
    private Tensor? _fcW, _fcB;               // acoustic_decoder.fc   : 2048 → 1024
    private Tensor? _embedW, _embedB;         // Conv1d 1024→1024 k7
    private readonly ResnetWeights[] _prior;
    private readonly ResnetWeights[] _post;
    private readonly TxWeights[] _blocks;
    private Tensor? _finalNormW, _finalNormB; // acoustic_decoder.norm (LayerNorm)
    private Tensor? _headW, _headB;           // head.linear : 1024 → n_fft+2
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

    /// <summary>Loads the decode-side weights straight from the checkpoint dictionary (raw
    /// safetensors key names, no prefix rewriting). The encode-side stacks
    /// (<c>acoustic_encoder</c>, <c>semantic_encoder</c>, <c>semantic_adapter</c>, <c>fc_encoder</c>,
    /// <c>quantizer.project_in</c>) are not needed for token→audio and are ignored.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "acoustic_decoder")
    {
        _projOutW = WhisperOps.EnsureF32(w["quantizer.project_out.weight"]);
        _projOutB = TryGet(w, "quantizer.project_out.bias");

        _fcW = WhisperOps.EnsureF32(w[$"{prefix}.fc.weight"]);
        _fcB = TryGet(w, $"{prefix}.fc.bias");

        _embedW = ConvW(w[$"{prefix}.embed.weight"], _cfg.BackboneDim, _cfg.BackboneDim, _cfg.EmbedKernel);
        _embedB = TryGet(w, $"{prefix}.embed.bias");

        for (int i = 0; i < _prior.Length; i++) _prior[i] = LoadResnet(w, $"{prefix}.prior_net.{i}");
        for (int i = 0; i < _post.Length; i++) _post[i] = LoadResnet(w, $"{prefix}.post_net.{i}");
        for (int i = 0; i < _blocks.Length; i++) _blocks[i] = LoadTx(w, $"{prefix}.layers.{i}");

        _finalNormW = WhisperOps.EnsureF32(w[$"{prefix}.norm.weight"]);
        _finalNormB = WhisperOps.EnsureF32(w[$"{prefix}.norm.bias"]);
        _headW = WhisperOps.EnsureF32(w[$"{prefix}.head.linear.weight"]);
        _headB = TryGet(w, $"{prefix}.head.linear.bias");
    }

    /// <summary>Decodes a single-codebook code stream <c>[F]</c> (FSQ indices) to 24 kHz mono PCM.</summary>
    public float[] Decode(IBackend backend, ReadOnlySpan<int> codes)
    {
        int f = codes.Length;
        int dim = _cfg.BackboneDim;

        // FSQ index → 8-d code in [-1,1] via the codebook rule (level - L/2) / (L/2) → project_out → fc → [1,F,1024].
        Tensor fsq = new(new TensorShape(1, f, _cfg.FsqDim), DType.F32);
        CodebookLookup(fsq, codes, f);
        Tensor q = WhisperOps.ProjectLinear(backend, fsq, _projOutW!, _projOutB, 1, f, _cfg.FsqDim, _cfg.QuantizerDim);
        fsq.Dispose();
        Tensor xCl = WhisperOps.ProjectLinear(backend, q, _fcW!, _fcB, 1, f, _cfg.QuantizerDim, dim);
        q.Dispose();

        // embed: Conv1d (channels-first), then prior ResNet pre-net.
        Tensor xCf = ToChannelsFirst(backend, xCl, f, dim); xCl.Dispose();
        Tensor emb = Conv1d(backend, xCf, _embedW!, _embedB, dim, dim, f, _cfg.EmbedKernel); xCf.Dispose();
        foreach (ResnetWeights r in _prior) { Tensor n = Resnet(backend, emb, r, dim, f); emb.Dispose(); emb = n; }

        // Transformer stack (channels-last, no positional encoding — see class remarks).
        Tensor t = ToChannelsLast(backend, emb, dim, f); emb.Dispose();
        foreach (TxWeights b in _blocks) { Tensor n = TxBlock(backend, t, b, f); t.Dispose(); t = n; }

        // Post ResNet (channels-first), then final LayerNorm (channels-last).
        Tensor pCf = ToChannelsFirst(backend, t, f, dim); t.Dispose();
        foreach (ResnetWeights r in _post) { Tensor n = Resnet(backend, pCf, r, dim, f); pCf.Dispose(); pCf = n; }
        Tensor headIn = ToChannelsLast(backend, pCf, dim, f); pCf.Dispose();
        Tensor ln = new(headIn.Shape, DType.F32);
        backend.LayerNorm(ln, headIn, _finalNormW!, _finalNormB!, _cfg.NormEps); headIn.Dispose();

        // Head: Linear(1024 → n_fft+2) → split mag/phase → "same"-padding iSTFT.
        int outDim = _cfg.NFft + 2;
        Tensor head = WhisperOps.ProjectLinear(backend, ln, _headW!, _headB, 1, f, dim, outDim);
        ln.Dispose();
        int halfBins = (_cfg.NFft / 2) + 1;
        float* hp = (float*)head.DataPointer;
        float[] re = new float[(long)f * halfBins];
        float[] im = new float[(long)f * halfBins];
        for (int frame = 0; frame < f; frame++)
        {
            int src = frame * outDim, dst = frame * halfBins;
            for (int k = 0; k < halfBins; k++)
            {
                float mag = MathF.Exp(hp[src + k]);
                if (mag > 100f) mag = 100f;
                float ph = hp[src + halfBins + k];
                re[dst + k] = mag * MathF.Cos(ph);
                im[dst + k] = mag * MathF.Sin(ph);
            }
        }
        head.Dispose();
        int edgePad = (_cfg.NFft - _cfg.HopLength) / 2;   // Vocos "same" padding → f * hop samples.
        return IStft.Apply(re, im, f, _cfg.NFft, _cfg.HopLength, edgePad);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_projOutW, _projOutB, _fcW, _fcB, _embedW, _embedB, _finalNormW, _finalNormB, _headW, _headB];
        foreach (Tensor? x in top) if (x is not null) yield return x;
        foreach (ResnetWeights r in _prior) foreach (Tensor x in r.All()) yield return x;
        foreach (ResnetWeights r in _post) foreach (Tensor x in r.All()) yield return x;
        foreach (TxWeights b in _blocks) foreach (Tensor x in b.All()) yield return x;
    }

    // ── FSQ codebook de-quant: index → per-axis level → (level - L/2) / (L/2), packed base-L, dim 0 least-significant ──
    private void CodebookLookup(Tensor dst, ReadOnlySpan<int> codes, int f)
    {
        int d = _cfg.FsqDim;
        Span<int> place = stackalloc int[d];
        place[0] = 1;
        for (int i = 1; i < d; i++) place[i] = place[i - 1] * _cfg.FsqLevels[i - 1];
        float* zp = (float*)dst.DataPointer;
        for (int ti = 0; ti < f; ti++)
        {
            int code = codes[ti];
            int rowBase = ti * d;
            for (int i = 0; i < d; i++)
            {
                int level = _cfg.FsqLevels[i];
                int halfWidth = level / 2;
                int digit = (code / place[i]) % level;
                zp[rowBase + i] = (digit - halfWidth) / (float)halfWidth;
            }
        }
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

    // ── Transformer block: RMSNorm + full MHA (separate q/k/v/o, no RoPE) + RMSNorm + SiLU MLP (channels-last [1,T,dim]) ──
    private Tensor TxBlock(IBackend backend, Tensor x, TxWeights b, int t)
    {
        int dim = _cfg.BackboneDim, heads = _cfg.NumHeads, hd = _cfg.HeadDim;
        Tensor pre = new(x.Shape, DType.F32);
        backend.RmsNorm(pre, x, b.AttNormW!, _cfg.NormEps);

        Tensor qFlat = WhisperOps.ProjectLinear(backend, pre, b.QW!, null, 1, t, dim, heads * hd);
        Tensor kFlat = WhisperOps.ProjectLinear(backend, pre, b.KW!, null, 1, t, dim, heads * hd);
        Tensor vFlat = WhisperOps.ProjectLinear(backend, pre, b.VW!, null, 1, t, dim, heads * hd); pre.Dispose();
        Tensor qMh = new(new TensorShape(1, heads, t, hd), DType.F32);
        Tensor kMh = new(new TensorShape(1, heads, t, hd), DType.F32);
        Tensor vMh = new(new TensorShape(1, heads, t, hd), DType.F32);
        FlatToHeads(qFlat, qMh, t, heads, hd); qFlat.Dispose();
        FlatToHeads(kFlat, kMh, t, heads, hd); kFlat.Dispose();
        FlatToHeads(vFlat, vMh, t, heads, hd); vFlat.Dispose();

        Tensor attn = new(new TensorShape(1, heads, t, hd), DType.F32);
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, null, 1f / MathF.Sqrt(hd));
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();
        Tensor attnFlat = new(new TensorShape(1, t, dim), DType.F32);
        HeadsToFlat(attn, attnFlat, t, heads, hd); attn.Dispose();
        Tensor attnOut = WhisperOps.ProjectLinear(backend, attnFlat, b.OW!, null, 1, t, dim, dim); attnFlat.Dispose();

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

    // Flat [1,T,heads*hd] (channel = h*hd + d) → heads [1,heads,T,hd].
    private static void FlatToHeads(Tensor flat, Tensor heads4, int t, int heads, int hd)
    {
        float* src = (float*)flat.DataPointer;
        float* dst = (float*)heads4.DataPointer;
        int dim = heads * hd;
        for (int s = 0; s < t; s++)
            for (int h = 0; h < heads; h++)
            {
                long srcOff = (long)s * dim + (long)h * hd;
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
        AttNormW = WhisperOps.EnsureF32(w[$"{p}.input_layernorm.weight"]),
        QW = WhisperOps.EnsureF32(w[$"{p}.self_attn.q_proj.weight"]),
        KW = WhisperOps.EnsureF32(w[$"{p}.self_attn.k_proj.weight"]),
        VW = WhisperOps.EnsureF32(w[$"{p}.self_attn.v_proj.weight"]),
        OW = WhisperOps.EnsureF32(w[$"{p}.self_attn.o_proj.weight"]),
        FfnNormW = WhisperOps.EnsureF32(w[$"{p}.post_attention_layernorm.weight"]),
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
        public Tensor? AttNormW, QW, KW, VW, OW, FfnNormW, Fc1W, Fc2W;
        public IEnumerable<Tensor> All()
        {
            Tensor?[] a = [AttNormW, QW, KW, VW, OW, FfnNormW, Fc1W, Fc2W];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }
}
