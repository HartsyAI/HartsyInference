using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>HunyuanVideo text-token refiner (<c>context_embedder</c> in diffusers
/// <c>HunyuanVideoTokenRefiner</c> / <c>txt_in</c> in the Tencent/Comfy checkpoint). Projects the primary
/// Llava-Llama-3 text hidden states <c>[B, L, textDim]</c> to the model width and passes them through
/// <c>numBlocks</c> (=2) timestep-gated self-attention refiner blocks, producing the text stream the DiT's
/// double/single blocks consume. Mirrors <c>transformer_hunyuan_video.py:HunyuanVideoTokenRefiner.forward</c>:
/// <list type="number">
/// <item>pooled = mean over the sequence of the raw text (unmasked; the pipeline crops to real tokens so no
/// padding mask is needed);</item>
/// <item><c>temb = t_embedder(sinusoid(timestep)) + c_embedder(pooled)</c> (both Linear→SiLU→Linear);</item>
/// <item><c>x = input_embedder(text)</c>;</item>
/// <item>each block: <c>x = x + proj(attn(LN(x))) · gate_msa</c> then <c>x = x + mlp(LN(x)) · gate_mlp</c>, where
/// <c>[gate_msa, gate_mlp] = adaLN(SiLU(temb))</c>.</item>
/// </list>
/// The attention is a plain multi-head self-attention (bias, <b>no</b> QK-norm, <b>no</b> RoPE). Weights are read
/// under the Tencent/Comfy <c>txt_in.*</c> naming (fused <c>self_attn.qkv</c>, split at load), so the checkpoint
/// converter carries them through 1:1.</summary>
public sealed unsafe class HunyuanVideoTokenRefiner
{
    private readonly int _textDim;
    private readonly int _hidden;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _numBlocks;
    private readonly int _mlpDim;

    private Tensor? _inW, _inB;             // input_embedder [hidden, textDim]
    private Tensor? _tW0, _tB0, _tW1, _tB1; // t_embedder [hidden,256]/[hidden,hidden]
    private Tensor? _cW0, _cB0, _cW1, _cB1; // c_embedder  [hidden,textDim]/[hidden,hidden]
    private readonly RefinerBlock[] _blocks;

    public HunyuanVideoTokenRefiner(int textDim, int hidden, int numHeads, int numBlocks = 2, float mlpRatio = 4f)
    {
        _textDim = textDim;
        _hidden = hidden;
        _numHeads = numHeads;
        _headDim = hidden / numHeads;
        _numBlocks = numBlocks;
        _mlpDim = (int)(hidden * mlpRatio);
        _blocks = new RefinerBlock[numBlocks];
        for (int i = 0; i < numBlocks; i++) _blocks[i] = new RefinerBlock(hidden, numHeads, _headDim, _mlpDim);
    }

    /// <summary>Loads weights under <paramref name="prefix"/> (default <c>txt_in</c>). Fused <c>self_attn.qkv</c> is split into per-projection Q/K/V at load. All tensors are cast to F32 (the refiner is small and stays resident).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "txt_in")
    {
        string p = prefix + ".";
        _inW = TensorCasts.EnsureF32(w[$"{p}input_embedder.weight"]); _inB = TensorCasts.EnsureF32(w[$"{p}input_embedder.bias"]);
        _tW0 = TensorCasts.EnsureF32(w[$"{p}t_embedder.in_layer.weight"]); _tB0 = TensorCasts.EnsureF32(w[$"{p}t_embedder.in_layer.bias"]);
        _tW1 = TensorCasts.EnsureF32(w[$"{p}t_embedder.out_layer.weight"]); _tB1 = TensorCasts.EnsureF32(w[$"{p}t_embedder.out_layer.bias"]);
        _cW0 = TensorCasts.EnsureF32(w[$"{p}c_embedder.in_layer.weight"]); _cB0 = TensorCasts.EnsureF32(w[$"{p}c_embedder.in_layer.bias"]);
        _cW1 = TensorCasts.EnsureF32(w[$"{p}c_embedder.out_layer.weight"]); _cB1 = TensorCasts.EnsureF32(w[$"{p}c_embedder.out_layer.bias"]);
        for (int i = 0; i < _numBlocks; i++)
            _blocks[i].LoadWeights(w, $"{p}individual_token_refiner.blocks.{i}", _hidden);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_inW, _inB, _tW0, _tB0, _tW1, _tB1, _cW0, _cB0, _cW1, _cB1];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        foreach (RefinerBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Refines <paramref name="text"/> <c>[B, L, textDim]</c> → <c>[B, L, hidden]</c> conditioned on <paramref name="timestep"/> (the scheduler timestep, ≈0..1000, same value the main DiT uses).</summary>
    public Tensor Forward(IBackend backend, Tensor text, float timestep)
    {
        int b = (int)text.Shape[0];
        int L = (int)text.Shape[1];

        // temb = t_embedder(sinusoid(t)) + c_embedder(mean_pool(text)).
        Tensor sin = new(new TensorShape(b, 256), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sin, timestep, b, 256);
        Tensor t0 = new(new TensorShape(b, _hidden), DType.F32); backend.Linear(t0, sin, _tW0!, _tB0!); sin.Dispose();
        Tensor t0a = new(t0.Shape, DType.F32); backend.Silu(t0a, t0); t0.Dispose();
        Tensor tEmb = new(new TensorShape(b, _hidden), DType.F32); backend.Linear(tEmb, t0a, _tW1!, _tB1!); t0a.Dispose();

        Tensor pooled = MeanPool(text, b, L, _textDim);
        Tensor c0 = new(new TensorShape(b, _hidden), DType.F32); backend.Linear(c0, pooled, _cW0!, _cB0!); pooled.Dispose();
        Tensor c0a = new(c0.Shape, DType.F32); backend.Silu(c0a, c0); c0.Dispose();
        Tensor cEmb = new(new TensorShape(b, _hidden), DType.F32); backend.Linear(cEmb, c0a, _cW1!, _cB1!); c0a.Dispose();

        Tensor temb = new(new TensorShape(b, _hidden), DType.F32); backend.Add(temb, tEmb, cEmb);
        tEmb.Dispose(); cEmb.Dispose();

        // x = input_embedder(text).
        Tensor x = new(new TensorShape(b, L, _hidden), DType.F32);
        backend.Linear(x, text, _inW!, _inB!);

        for (int i = 0; i < _numBlocks; i++)
        {
            Tensor nx = _blocks[i].Forward(backend, x, temb, b, L);
            x.Dispose(); x = nx;
        }
        temb.Dispose();
        return x;
    }

    /// <summary>Mean over the sequence dim: <c>[B, L, D] → [B, D]</c>.</summary>
    private static Tensor MeanPool(Tensor text, int b, int L, int d)
    {
        Tensor outT = new(new TensorShape(b, d), DType.F32);
        float* tp = (float*)text.DataPointer; float* op = (float*)outT.DataPointer;
        float inv = 1f / L;
        for (int bb = 0; bb < b; bb++)
            for (int dd = 0; dd < d; dd++)
            {
                double s = 0;
                for (int l = 0; l < L; l++) s += tp[((long)bb * L + l) * d + dd];
                op[(long)bb * d + dd] = (float)(s * inv);
            }
        return outT;
    }

    /// <summary>One <c>TokenRefinerBlock</c>: LN(affine) → MHA self-attn (bias, no QK-norm/RoPE) → gated residual → LN(affine) → Linear→SiLU→Linear MLP → gated residual. The two gates come from <c>adaLN_modulation.1(SiLU(temb)).chunk(2)</c>.</summary>
    private sealed class RefinerBlock
    {
        private readonly int _hidden, _numHeads, _headDim, _mlpDim;
        private Tensor? _n1W, _n1B, _n2W, _n2B;
        private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _projW, _projB;
        private Tensor? _mlp0W, _mlp0B, _mlp2W, _mlp2B;
        private Tensor? _modW, _modB;   // adaLN_modulation.1 [2*hidden, hidden]

        public RefinerBlock(int hidden, int numHeads, int headDim, int mlpDim)
        { _hidden = hidden; _numHeads = numHeads; _headDim = headDim; _mlpDim = mlpDim; }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, int hidden)
        {
            string p = prefix + ".";
            _n1W = TensorCasts.EnsureF32(w[$"{p}norm1.weight"]); _n1B = TensorCasts.EnsureF32(w[$"{p}norm1.bias"]);
            _n2W = TensorCasts.EnsureF32(w[$"{p}norm2.weight"]); _n2B = TensorCasts.EnsureF32(w[$"{p}norm2.bias"]);
            // Fused qkv [3*hidden, hidden] → split into Q/K/V.
            Tensor qkvW = TensorCasts.EnsureF32(w[$"{p}self_attn.qkv.weight"]);
            Tensor qkvB = TensorCasts.EnsureF32(w[$"{p}self_attn.qkv.bias"]);
            (_qW, _kW, _vW) = SplitRows3(qkvW, hidden);
            (_qB, _kB, _vB) = SplitRows3(qkvB, hidden);
            _projW = TensorCasts.EnsureF32(w[$"{p}self_attn.proj.weight"]); _projB = TensorCasts.EnsureF32(w[$"{p}self_attn.proj.bias"]);
            _mlp0W = TensorCasts.EnsureF32(w[$"{p}mlp.0.weight"]); _mlp0B = TensorCasts.EnsureF32(w[$"{p}mlp.0.bias"]);
            _mlp2W = TensorCasts.EnsureF32(w[$"{p}mlp.2.weight"]); _mlp2B = TensorCasts.EnsureF32(w[$"{p}mlp.2.bias"]);
            _modW = TensorCasts.EnsureF32(w[$"{p}adaLN_modulation.1.weight"]); _modB = TensorCasts.EnsureF32(w[$"{p}adaLN_modulation.1.bias"]);
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor? t in new[] { _n1W, _n1B, _n2W, _n2B, _qW, _qB, _kW, _kB, _vW, _vB,
                _projW, _projB, _mlp0W, _mlp0B, _mlp2W, _mlp2B, _modW, _modB })
                if (t is not null) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor x, Tensor temb, int b, int L)
        {
            // gates = adaLN(SiLU(temb)) → [B, 2*hidden].
            Tensor tAct = new(temb.Shape, DType.F32); backend.Silu(tAct, temb);
            Tensor mod = new(new TensorShape(b, 2 * _hidden), DType.F32); backend.Linear(mod, tAct, _modW!, _modB!); tAct.Dispose();
            (Tensor gateMsa, Tensor gateMlp) = SplitCols2(mod, b, _hidden); mod.Dispose();

            // Self-attention.
            Tensor norm1 = LayerNormAffine(x, _n1W!, _n1B!, b, L, _hidden);
            TensorShape sh = new(b, L, _hidden);
            Tensor q = new(sh, DType.F32); backend.Linear(q, norm1, _qW!, _qB!);
            Tensor k = new(sh, DType.F32); backend.Linear(k, norm1, _kW!, _kB!);
            Tensor v = new(sh, DType.F32); backend.Linear(v, norm1, _vW!, _vB!);
            norm1.Dispose();

            TensorShape mh = new(b, _numHeads, L, _headDim);
            Tensor qMh = new(mh, DType.F32), kMh = new(mh, DType.F32), vMh = new(mh, DType.F32);
            DiTUtils.ReshapeToMultiHead(qMh, q, b, L, _numHeads, _headDim); q.Dispose();
            DiTUtils.ReshapeToMultiHead(kMh, k, b, L, _numHeads, _headDim); k.Dispose();
            DiTUtils.ReshapeToMultiHead(vMh, v, b, L, _numHeads, _headDim); v.Dispose();

            Tensor attnMh = new(mh, DType.F32);
            backend.ScaledDotProductAttention(attnMh, qMh, kMh, vMh, null, 1f / MathF.Sqrt(_headDim));
            qMh.Dispose(); kMh.Dispose(); vMh.Dispose();
            Tensor attnFlat = new(sh, DType.F32); DiTUtils.ReshapeFromMultiHead(attnFlat, attnMh, b, L, _numHeads, _headDim); attnMh.Dispose();
            Tensor attn = new(sh, DType.F32); backend.Linear(attn, attnFlat, _projW!, _projB!); attnFlat.Dispose();

            Tensor x1 = AdaLNModulation.ApplyGatedResidual(x, attn, gateMsa, b, L, _hidden);
            attn.Dispose(); gateMsa.Dispose();

            // MLP.
            Tensor norm2 = LayerNormAffine(x1, _n2W!, _n2B!, b, L, _hidden);
            Tensor m0 = new(new TensorShape(b, L, _mlpDim), DType.F32); backend.Linear(m0, norm2, _mlp0W!, _mlp0B!); norm2.Dispose();
            Tensor m0a = new(m0.Shape, DType.F32); backend.Silu(m0a, m0); m0.Dispose();
            Tensor m2 = new(sh, DType.F32); backend.Linear(m2, m0a, _mlp2W!, _mlp2B!); m0a.Dispose();

            Tensor x2 = AdaLNModulation.ApplyGatedResidual(x1, m2, gateMlp, b, L, _hidden);
            m2.Dispose(); gateMlp.Dispose(); x1.Dispose();
            return x2;
        }

        /// <summary>Affine LayerNorm over the last dim (eps 1e-6): <c>(x-μ)/σ · w + b</c>.</summary>
        private static Tensor LayerNormAffine(Tensor x, Tensor weight, Tensor bias, int b, int L, int d)
        {
            Tensor outT = new(x.Shape, DType.F32);
            float* xp = (float*)x.DataPointer; float* op = (float*)outT.DataPointer;
            float* wp = (float*)weight.DataPointer; float* bp = (float*)bias.DataPointer;
            for (long row = 0; row < (long)b * L; row++)
            {
                float* xr = xp + row * d; float* or_ = op + row * d;
                double mean = 0; for (int c = 0; c < d; c++) mean += xr[c]; mean /= d;
                double var = 0; for (int c = 0; c < d; c++) { double dd = xr[c] - mean; var += dd * dd; } var /= d;
                float inv = (float)(1.0 / Math.Sqrt(var + 1e-6));
                for (int c = 0; c < d; c++) or_[c] = ((float)(xr[c] - mean)) * inv * wp[c] + bp[c];
            }
            return outT;
        }

        private static (Tensor Q, Tensor K, Tensor V) SplitRows3(Tensor fused, int hidden)
        {
            // fused rows = [Q(hidden) | K(hidden) | V(hidden)]; supports 2D [3h,in] and 1D [3h] (bias).
            int cols = fused.Shape.Rank == 2 ? (int)fused.Shape[1] : 1;
            Tensor q = Make(hidden, cols), k = Make(hidden, cols), v = Make(hidden, cols);
            float* sp = (float*)fused.DataPointer;
            Copy(sp + 0L * hidden * cols, q, (long)hidden * cols);
            Copy(sp + 1L * hidden * cols, k, (long)hidden * cols);
            Copy(sp + 2L * hidden * cols, v, (long)hidden * cols);
            return (q, k, v);

            static Tensor Make(int rows, int cols) => cols == 1
                ? new Tensor(new TensorShape(rows), DType.F32)
                : new Tensor(new TensorShape(rows, cols), DType.F32);
            static void Copy(float* src, Tensor dst, long n)
            { float* dp = (float*)dst.DataPointer; Buffer.MemoryCopy(src, dp, n * 4, n * 4); }
        }

        private static (Tensor A, Tensor B) SplitCols2(Tensor mod, int b, int hidden)
        {
            Tensor a = new(new TensorShape(b, hidden), DType.F32), bb = new(new TensorShape(b, hidden), DType.F32);
            float* sp = (float*)mod.DataPointer; float* ap = (float*)a.DataPointer; float* bp = (float*)bb.DataPointer;
            for (int i = 0; i < b; i++)
            {
                Buffer.MemoryCopy(sp + (long)i * 2 * hidden, ap + (long)i * hidden, (long)hidden * 4, (long)hidden * 4);
                Buffer.MemoryCopy(sp + (long)i * 2 * hidden + hidden, bp + (long)i * hidden, (long)hidden * 4, (long)hidden * 4);
            }
            return (a, bb);
        }
    }
}
