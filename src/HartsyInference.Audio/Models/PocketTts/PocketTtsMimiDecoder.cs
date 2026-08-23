using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.PocketTts;

/// <summary>Pocket-TTS continuous-latent Mimi DECODE path (latent[32] -> 24 kHz audio), matching the
/// kyutai-labs/pocket-tts reference (moshi-native, pre-fused weight keys). Pipeline:
/// <list type="number">
///   <item><c>quantizer.output_proj</c>: 1x1 conv 32 -> 512.</item>
///   <item><c>upsample</c>: depthwise <c>ConvTranspose1d</c> (groups=512, k=32, stride=16), causal trim
///         (drop last <c>k-stride=16</c>) -> 12.5 Hz to 200 Hz.</item>
///   <item><c>decoder_transformer</c>: 2-layer ProjectedTransformer (pre-LayerNorm, fused-QKV attention with
///         interleaved RoPE + sliding-window <c>context=250</c>, LayerScale 0.01, bias-free exact-GELU MLP).</item>
///   <item><c>decoder</c>: moshi SEANet decoder (ratios [6,5,4], n_filters 64, last_kernel 3, causal convs,
///         ELU) -> 200 Hz to 24 kHz.</item>
/// </list>
/// All convs are causal (left-pad <c>k_eff-stride</c> zeros; transposed convs trim the trailing <c>k-stride</c>),
/// reproducing the streaming modules run with a fresh state. Verified by <c>SnacParityTests</c>-style oracle.</summary>
internal sealed unsafe class PocketTtsMimiDecoder
{
    private const int LatentDim = 32;
    private const int Dim = 512;          // outer_dim / d_model
    private const int NHeads = 8;
    private const int HeadDim = Dim / NHeads;
    private const int NLayers = 2;
    private const int Ffn = 2048;
    private const int Context = 250;
    private const float LnEps = 1e-5f;
    private const float RopeMaxPeriod = 10_000f;
    private const int UpStride = 16;
    private const int UpKernel = 32;

    // SEANet decoder shape (config: ratios [6,5,4], n_filters 64, dimension 512).
    private static readonly int[] Ratios = [6, 5, 4];
    private const int NFilters = 64;
    private const int LastKernel = 3;
    private const int ResidualKernel = 3;

    private readonly string _p;

    // Front: output_proj + upsample.
    private Tensor? _outProjW;     // [512,32,1]
    private Tensor? _upW;          // [512,1,32] depthwise convtr

    // Transformer (per layer).
    private readonly Tensor?[] _n1W = new Tensor?[NLayers], _n1B = new Tensor?[NLayers];
    private readonly Tensor?[] _n2W = new Tensor?[NLayers], _n2B = new Tensor?[NLayers];
    private readonly Tensor?[] _inProjW = new Tensor?[NLayers];   // [1536,512]
    private readonly Tensor?[] _outProjAttnW = new Tensor?[NLayers]; // [512,512]
    private readonly Tensor?[] _lin1W = new Tensor?[NLayers];     // [2048,512]
    private readonly Tensor?[] _lin2W = new Tensor?[NLayers];     // [512,2048]
    private readonly Tensor?[] _ls1 = new Tensor?[NLayers], _ls2 = new Tensor?[NLayers];

    // SEANet decoder weights (moshi indices; ELU layers occupy slots 1,4,7,10).
    private Tensor? _seInW, _seInB;        // model.0 conv  512->512 k7
    private readonly Tensor?[] _seUpW = new Tensor?[3], _seUpB = new Tensor?[3];     // model.2/5/8 convtr
    private readonly Tensor?[] _seR1W = new Tensor?[3], _seR1B = new Tensor?[3];     // resblock.block.1 conv (k3)
    private readonly Tensor?[] _seR2W = new Tensor?[3], _seR2B = new Tensor?[3];     // resblock.block.3 conv (k1)
    private Tensor? _seOutW, _seOutB;      // model.11 conv 64->1 k3

    public PocketTtsMimiDecoder(string prefix = "mimi")
    {
        _p = prefix;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _outProjW = WhisperOps.EnsureF32(w[$"{_p}.quantizer.output_proj.weight"]);
        _upW = WhisperOps.EnsureF32(w[$"{_p}.upsample.convtr.convtr.weight"]);

        for (int i = 0; i < NLayers; i++)
        {
            string p = $"{_p}.decoder_transformer.transformer.layers.{i}";
            _n1W[i] = WhisperOps.EnsureF32(w[$"{p}.norm1.weight"]);
            _n1B[i] = WhisperOps.EnsureF32(w[$"{p}.norm1.bias"]);
            _n2W[i] = WhisperOps.EnsureF32(w[$"{p}.norm2.weight"]);
            _n2B[i] = WhisperOps.EnsureF32(w[$"{p}.norm2.bias"]);
            _inProjW[i] = WhisperOps.EnsureF32(w[$"{p}.self_attn.in_proj.weight"]);
            _outProjAttnW[i] = WhisperOps.EnsureF32(w[$"{p}.self_attn.out_proj.weight"]);
            _lin1W[i] = WhisperOps.EnsureF32(w[$"{p}.linear1.weight"]);
            _lin2W[i] = WhisperOps.EnsureF32(w[$"{p}.linear2.weight"]);
            _ls1[i] = WhisperOps.EnsureF32(w[$"{p}.layer_scale_1.scale"]);
            _ls2[i] = WhisperOps.EnsureF32(w[$"{p}.layer_scale_2.scale"]);
        }

        string d = $"{_p}.decoder.model";
        _seInW = WhisperOps.EnsureF32(w[$"{d}.0.conv.weight"]);
        _seInB = WhisperOps.EnsureF32(w[$"{d}.0.conv.bias"]);
        int[] upIdx = [2, 5, 8];
        int[] rbIdx = [3, 6, 9];
        for (int s = 0; s < 3; s++)
        {
            _seUpW[s] = WhisperOps.EnsureF32(w[$"{d}.{upIdx[s]}.convtr.weight"]);
            _seUpB[s] = WhisperOps.EnsureF32(w[$"{d}.{upIdx[s]}.convtr.bias"]);
            _seR1W[s] = WhisperOps.EnsureF32(w[$"{d}.{rbIdx[s]}.block.1.conv.weight"]);
            _seR1B[s] = WhisperOps.EnsureF32(w[$"{d}.{rbIdx[s]}.block.1.conv.bias"]);
            _seR2W[s] = WhisperOps.EnsureF32(w[$"{d}.{rbIdx[s]}.block.3.conv.weight"]);
            _seR2B[s] = WhisperOps.EnsureF32(w[$"{d}.{rbIdx[s]}.block.3.conv.bias"]);
        }
        _seOutW = WhisperOps.EnsureF32(w[$"{d}.11.conv.weight"]);
        _seOutB = WhisperOps.EnsureF32(w[$"{d}.11.conv.bias"]);
    }

    /// <summary>latent <c>[1, 32, T]</c> -> pcm <c>[1, 1, T*1920]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, int batch, int tFrames)
    {
        // 1) output_proj 1x1: 32 -> 512.
        Tensor emb = new(new TensorShape(batch, Dim, tFrames), DType.F32);
        backend.Conv1d(emb, latent, _outProjW!, null, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);

        // 2) depthwise upsample (k32, s16) with causal trim (padRight = k - s = 16).
        int tUp = tFrames * UpStride;
        Tensor up = new(new TensorShape(batch, Dim, tUp), DType.F32);
        backend.ConvTranspose1d(up, emb, _upW!, null, stride: UpStride, padLeft: 0, padRight: UpKernel - UpStride, dilation: 1, groups: Dim);
        emb.Dispose();

        // 3) decoder transformer (channels-last).
        Tensor xCl = new(new TensorShape(batch, tUp, Dim), DType.F32);
        backend.Transpose2D(xCl, up, Dim, tUp);
        up.Dispose();
        Tensor ctx = RunTransformer(backend, xCl, batch, tUp);
        xCl.Dispose();
        Tensor ctxCf = new(new TensorShape(batch, Dim, tUp), DType.F32);
        backend.Transpose2D(ctxCf, ctx, tUp, Dim);
        ctx.Dispose();

        // 4) SEANet decoder.
        Tensor pcm = RunSeanet(backend, ctxCf, batch, tUp);
        ctxCf.Dispose();
        return pcm;
    }

    // ── decoder transformer ──
    private Tensor RunTransformer(IBackend backend, Tensor x, int batch, int t)
    {
        Tensor cur = new(x.Shape, DType.F32);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)cur.DataPointer, x.Shape.ElementCount * 4, x.Shape.ElementCount * 4);

        Tensor mask = BuildSlidingCausalMask(t, Context);
        for (int i = 0; i < NLayers; i++)
        {
            // self-attention block: cur + ls1 * out_proj(attn(LN1(cur)))
            Tensor normed = new(cur.Shape, DType.F32);
            backend.LayerNorm(normed, cur, _n1W[i]!, _n1B[i]!, LnEps);
            Tensor qkv = WhisperOps.ProjectLinear(backend, normed, _inProjW[i]!, null, batch, t, Dim, 3 * Dim);
            normed.Dispose();

            Tensor qMh = new(new TensorShape(batch, NHeads, t, HeadDim), DType.F32);
            Tensor kMh = new(new TensorShape(batch, NHeads, t, HeadDim), DType.F32);
            Tensor vMh = new(new TensorShape(batch, NHeads, t, HeadDim), DType.F32);
            SplitQkvToHeads(qkv, qMh, kMh, vMh, batch, t);
            qkv.Dispose();
            ApplyRopeInterleaved(qMh, batch, NHeads, t, HeadDim, RopeMaxPeriod);
            ApplyRopeInterleaved(kMh, batch, NHeads, t, HeadDim, RopeMaxPeriod);

            Tensor attn = new(qMh.Shape, DType.F32);
            backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, mask, 1f / MathF.Sqrt(HeadDim));
            qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

            Tensor attnFlat = new(new TensorShape(batch, t, Dim), DType.F32);
            MergeHeads(attn, attnFlat, batch, t);
            attn.Dispose();
            Tensor outp = WhisperOps.ProjectLinear(backend, attnFlat, _outProjAttnW[i]!, null, batch, t, Dim, Dim);
            attnFlat.Dispose();
            AddScaled(cur, outp, _ls1[i]!, batch, t);   // cur += ls1 * outp
            outp.Dispose();

            // feed-forward block: cur + ls2 * lin2(gelu(lin1(LN2(cur))))
            Tensor normed2 = new(cur.Shape, DType.F32);
            backend.LayerNorm(normed2, cur, _n2W[i]!, _n2B[i]!, LnEps);
            Tensor h = WhisperOps.ProjectLinear(backend, normed2, _lin1W[i]!, null, batch, t, Dim, Ffn);
            normed2.Dispose();
            Activations.ErfGelu(h);
            Tensor ff = WhisperOps.ProjectLinear(backend, h, _lin2W[i]!, null, batch, t, Ffn, Dim);
            h.Dispose();
            AddScaled(cur, ff, _ls2[i]!, batch, t);
            ff.Dispose();
        }
        mask.Dispose();
        return cur;
    }

    private static void SplitQkvToHeads(Tensor qkv, Tensor q, Tensor k, Tensor v, int batch, int t)
    {
        // qkv [B,T,3*Dim] packed (q|k|v); each -> [B,H,T,D]
        float* s = (float*)qkv.DataPointer;
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int ti = 0; ti < t; ti++)
            {
                long rb = ((long)b * t + ti) * 3 * Dim;
                for (int h = 0; h < NHeads; h++)
                    for (int dd = 0; dd < HeadDim; dd++)
                    {
                        long dst = (((long)b * NHeads + h) * t + ti) * HeadDim + dd;
                        int c = h * HeadDim + dd;
                        qp[dst] = s[rb + c];
                        kp[dst] = s[rb + Dim + c];
                        vp[dst] = s[rb + 2 * Dim + c];
                    }
            }
    }

    private static void MergeHeads(Tensor attn, Tensor flat, int batch, int t)
    {
        float* ap = (float*)attn.DataPointer, fp = (float*)flat.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int ti = 0; ti < t; ti++)
                for (int h = 0; h < NHeads; h++)
                    for (int dd = 0; dd < HeadDim; dd++)
                        fp[((long)b * t + ti) * Dim + h * HeadDim + dd]
                            = ap[(((long)b * NHeads + h) * t + ti) * HeadDim + dd];
    }

    private static void AddScaled(Tensor cur, Tensor upd, Tensor scale, int batch, int t)
    {
        float* cp = (float*)cur.DataPointer, up = (float*)upd.DataPointer, sp = (float*)scale.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int ti = 0; ti < t; ti++)
                for (int c = 0; c < Dim; c++)
                {
                    long idx = ((long)b * t + ti) * Dim + c;
                    cp[idx] += sp[c] * up[idx];
                }
    }

    /// <summary>Interleaved RoPE on <c>[B,H,T,D]</c>: pairs (2k, 2k+1), freq = maxPeriod^(-2k/D).</summary>
    private static void ApplyRopeInterleaved(Tensor x, int batch, int heads, int t, int headDim, float maxPeriod)
    {
        float* xp = (float*)x.DataPointer;
        int half = headDim / 2;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
                for (int ti = 0; ti < t; ti++)
                {
                    long rb = (((long)b * heads + h) * t + ti) * headDim;
                    for (int kk = 0; kk < half; kk++)
                    {
                        float freq = MathF.Exp(kk * (-MathF.Log(maxPeriod) * 2f / headDim));
                        float ang = ti * freq;
                        float cs = MathF.Cos(ang), sn = MathF.Sin(ang);
                        float r = xp[rb + 2 * kk], im = xp[rb + 2 * kk + 1];
                        xp[rb + 2 * kk] = r * cs - im * sn;
                        xp[rb + 2 * kk + 1] = r * sn + im * cs;
                    }
                }
    }

    private static Tensor BuildSlidingCausalMask(int t, int context)
    {
        Tensor mask = new(new TensorShape(t, t), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int q = 0; q < t; q++)
            for (int kk = 0; kk < t; kk++)
            {
                int delta = q - kk;
                bool ok = delta >= 0 && delta < context;
                mp[(long)q * t + kk] = ok ? 0f : float.NegativeInfinity;
            }
        return mask;
    }

    // ── SEANet decoder (causal) ──
    private Tensor RunSeanet(IBackend backend, Tensor z, int batch, int t)
    {
        // model.0: conv 512->512 k7 (causal padLeft 6)
        Tensor x = CausalConv(backend, z, _seInW!, _seInB!, dim: Dim, t: t, kernel: 7, dilation: 1);
        int curT = t;
        int mult = 1 << Ratios.Length;       // 8
        for (int s = 0; s < Ratios.Length; s++)
        {
            int inCh = mult * NFilters;
            int outCh = inCh / 2;
            int ratio = Ratios[s];
            Tensor e = new(x.Shape, DType.F32); backend.Elu(e, x, 1.0f); x.Dispose(); x = e;
            // convtr inCh->outCh, k=2r s=r, causal trim padRight=r
            int tUp = curT * ratio;
            Tensor up = new(new TensorShape(batch, outCh, tUp), DType.F32);
            backend.ConvTranspose1d(up, x, _seUpW[s]!, _seUpB[s], stride: ratio, padLeft: 0, padRight: 2 * ratio - ratio, dilation: 1, groups: 1);
            x.Dispose(); x = up; curT = tUp;
            // resnet block: v = ELU; conv k3 dil1 outCh->outCh/2... actually block conv1: outCh -> outCh/compress, conv2: ->outCh
            x = ResnetBlock(backend, x, _seR1W[s]!, _seR1B[s]!, _seR2W[s]!, _seR2B[s]!, outCh, curT, batch);
            mult /= 2;
        }
        // final ELU + conv 64->1 k3 (causal)
        Tensor ef = new(x.Shape, DType.F32); backend.Elu(ef, x, 1.0f); x.Dispose(); x = ef;
        Tensor pcm = CausalConvOut(backend, x, _seOutW!, _seOutB!, inDim: NFilters, outDim: 1, t: curT, kernel: LastKernel, batch: batch);
        x.Dispose();
        return pcm;
    }

    private Tensor ResnetBlock(IBackend backend, Tensor x, Tensor c1w, Tensor c1b, Tensor c2w, Tensor c2b, int dim, int t, int batch)
    {
        // block.0 ELU, block.1 conv (dim -> dim/compress, k3 dil1), block.2 ELU, block.3 conv (dim/compress -> dim, k1); residual add.
        int hidden = dim / 2;   // compress=2
        Tensor e1 = new(x.Shape, DType.F32); backend.Elu(e1, x, 1.0f);
        Tensor mid = CausalConvCh(backend, e1, c1w, c1b, dim, hidden, t, ResidualKernel, 1, batch);
        e1.Dispose();
        Tensor e2 = new(mid.Shape, DType.F32); backend.Elu(e2, mid, 1.0f); mid.Dispose();
        Tensor proj = CausalConvCh(backend, e2, c2w, c2b, hidden, dim, t, 1, 1, batch);
        e2.Dispose();
        Tensor outp = new(x.Shape, DType.F32);
        backend.Add(outp, x, proj);
        proj.Dispose(); x.Dispose();
        return outp;
    }

    // causal conv dim->dim same length
    private static Tensor CausalConv(IBackend backend, Tensor x, Tensor wt, Tensor b, int dim, int t, int kernel, int dilation)
    {
        int padLeft = (kernel - 1) * dilation;   // k_eff - stride, stride=1
        int batch = (int)x.Shape[0];
        Tensor o = new(new TensorShape(batch, dim, t), DType.F32);
        backend.Conv1d(o, x, wt, b, stride: 1, padLeft: padLeft, padRight: 0, dilation: dilation, groups: 1);
        return o;
    }

    private static Tensor CausalConvCh(IBackend backend, Tensor x, Tensor wt, Tensor b, int inDim, int outDim, int t, int kernel, int dilation, int batch)
    {
        int padLeft = (kernel - 1) * dilation;
        Tensor o = new(new TensorShape(batch, outDim, t), DType.F32);
        backend.Conv1d(o, x, wt, b, stride: 1, padLeft: padLeft, padRight: 0, dilation: dilation, groups: 1);
        return o;
    }

    private static Tensor CausalConvOut(IBackend backend, Tensor x, Tensor wt, Tensor b, int inDim, int outDim, int t, int kernel, int batch)
    {
        int padLeft = kernel - 1;
        Tensor o = new(new TensorShape(batch, outDim, t), DType.F32);
        backend.Conv1d(o, x, wt, b, stride: 1, padLeft: padLeft, padRight: 0, dilation: 1, groups: 1);
        return o;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _outProjW, _upW, _seInW, _seInB, _seOutW, _seOutB,
            .. _n1W, .. _n1B, .. _n2W, .. _n2B, .. _inProjW, .. _outProjAttnW, .. _lin1W, .. _lin2W, .. _ls1, .. _ls2,
            .. _seUpW, .. _seUpB, .. _seR1W, .. _seR1B, .. _seR2W, .. _seR2B,
        ];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
