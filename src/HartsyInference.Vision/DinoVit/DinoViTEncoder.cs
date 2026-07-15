using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.DinoVit;

/// <summary>HuggingFace <c>ViTModel</c> backbone (DINOv1 <c>facebook/dino-vitb16</c>) as an image tokenizer:
/// patch-embed conv → prepend CLS → add bicubic-interpolated position embeddings → pre-norm transformer
/// layers (exact-erf GELU MLP) → final LayerNorm. Returns the full token sequence <c>[B, 1+patches, hidden]</c>
/// (the <c>last_hidden_state</c>) that image→3D backbones cross-attend to. All matmuls route through
/// <see cref="IBackend"/>.</summary>
public sealed unsafe class DinoViTEncoder
{
    private readonly DinoViTConfig _cfg;
    private readonly DinoViTLayer[] _layers;

    private Tensor? _patchW, _patchB;   // projection.weight [hidden,3,p,p], bias [hidden]
    private Tensor? _clsToken;          // [1,1,hidden]
    private Tensor? _posEmb;            // [1, 1+nativePatches, hidden]
    private Tensor? _finalLnW, _finalLnB;
    private Tensor? _interpPos;         // cached [1, 1+patches, hidden] interpolated position embeddings

    public DinoViTConfig Config => _cfg;

    public DinoViTEncoder(DinoViTConfig cfg)
    {
        _cfg = cfg;
        _layers = new DinoViTLayer[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++)
            _layers[i] = new DinoViTLayer(cfg.HiddenSize, cfg.NumHeads, cfg.IntermediateSize, cfg.LayerNormEps);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _patchW = F32(w[$"{p}embeddings.patch_embeddings.projection.weight"]);
        _patchB = F32(w[$"{p}embeddings.patch_embeddings.projection.bias"]);
        _clsToken = F32(w[$"{p}embeddings.cls_token"]);
        _posEmb = F32(w[$"{p}embeddings.position_embeddings"]);
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{p}encoder.layer.{i}");
        _finalLnW = F32(w[$"{p}layernorm.weight"]);
        _finalLnB = F32(w[$"{p}layernorm.bias"]);
        _interpPos = InterpolatePosEmbeddings();
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_patchW, _patchB, _clsToken, _posEmb, _finalLnW, _finalLnB, _interpPos];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        foreach (DinoViTLayer l in _layers) foreach (Tensor t in l.EnumerateWeights()) yield return t;
    }

    /// <summary>Encodes preprocessed pixels <c>[B,3,ImageSize,ImageSize]</c> (already ImageNet-normalized)
    /// into the token sequence <c>[B, 1+patches, hidden]</c> after the final layer norm.</summary>
    public Tensor Encode(IBackend backend, Tensor pixelValues)
    {
        int batch = (int)pixelValues.Shape[0];
        int hidden = _cfg.HiddenSize, grid = _cfg.PatchGrid, numPatches = _cfg.NumPatches, seqLen = _cfg.SequenceLength;

        Tensor patch = new(new TensorShape(batch, hidden, grid, grid), DType.F32);
        backend.Conv2D(patch, pixelValues, _patchW!, _patchB!, _cfg.PatchSize, _cfg.PatchSize, 0, 0);
        Tensor flat = patch.Reshape(new TensorShape(batch, hidden, numPatches));
        Tensor patches = new(new TensorShape(batch, numPatches, hidden), DType.F32);
        backend.Transpose2D(patches, flat, hidden, numPatches);
        patch.Dispose();

        // [CLS, patches] + interpolated position embeddings.
        Tensor seq = new(new TensorShape(batch, seqLen, hidden), DType.F32);
        float* dp = (float*)seq.DataPointer;
        float* cls = (float*)_clsToken!.DataPointer;
        float* pos = (float*)_interpPos!.DataPointer;     // [1, seqLen, hidden]
        float* pat = (float*)patches.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < hidden; c++) dp[(long)(b * seqLen) * hidden + c] = cls[c] + pos[c];
            for (int pi = 0; pi < numPatches; pi++)
            {
                long dstBase = ((long)(b * seqLen) + 1 + pi) * hidden;
                long patBase = ((long)b * numPatches + pi) * hidden;
                long posBase = (long)(1 + pi) * hidden;
                for (int c = 0; c < hidden; c++) dp[dstBase + c] = pat[patBase + c] + pos[posBase + c];
            }
        }
        patches.Dispose();

        Tensor h = seq;
        for (int i = 0; i < _layers.Length; i++) { Tensor n = _layers[i].Forward(backend, h); h.Dispose(); h = n; }

        Tensor outp = new(h.Shape, DType.F32);
        backend.LayerNorm(outp, h, _finalLnW!, _finalLnB!, _cfg.LayerNormEps);
        h.Dispose();
        return outp;
    }

    /// <summary>Bicubic-interpolates the native patch position grid to the conditioning grid (PyTorch
    /// <c>F.interpolate(mode="bicubic", align_corners=False)</c>), then re-prepends the CLS position. Cached.</summary>
    private Tensor InterpolatePosEmbeddings()
    {
        int hidden = _cfg.HiddenSize, src = _cfg.NativePatchGrid, dst = _cfg.PatchGrid, seqLen = _cfg.SequenceLength;
        Tensor outp = new(new TensorShape(1, seqLen, hidden), DType.F32);
        float* op = (float*)outp.DataPointer;
        float* src_ = (float*)_posEmb!.DataPointer;       // [1, 1+src², hidden]
        // CLS position (row 0).
        for (int c = 0; c < hidden; c++) op[c] = src_[c];
        if (src == dst)
        {
            for (long i = 0; i < (long)(seqLen - 1) * hidden; i++) op[hidden + i] = src_[hidden + i];
            return outp;
        }
        // Patch positions start at offset `hidden` in src (after CLS). Interpolate per channel.
        // HF/DINO add a +0.1 fudge to the target grid and pass it to F.interpolate as a *scale_factor*
        // ((dst+0.1)/src). With align_corners=False + an explicit scale_factor, PyTorch's coordinate scale is
        // 1/scale_factor = src/(dst+0.1), NOT src/dst — the +0.1 shifts every sample and must be reproduced.
        float scale = src / (dst + 0.1f);
        Span<float> wx = stackalloc float[4];
        Span<float> wy = stackalloc float[4];
        for (int oy = 0; oy < dst; oy++)
        {
            float ry = scale * (oy + 0.5f) - 0.5f;
            int iy = (int)MathF.Floor(ry);
            CubicCoeffs(ry - iy, wy);
            for (int ox = 0; ox < dst; ox++)
            {
                float rx = scale * (ox + 0.5f) - 0.5f;
                int ix = (int)MathF.Floor(rx);
                CubicCoeffs(rx - ix, wx);
                long dstBase = (long)(1 + oy * dst + ox) * hidden;
                for (int c = 0; c < hidden; c++)
                {
                    float acc = 0f;
                    for (int ky = 0; ky < 4; ky++)
                    {
                        int sy = Math.Clamp(iy - 1 + ky, 0, src - 1);
                        float rowAcc = 0f;
                        for (int kx = 0; kx < 4; kx++)
                        {
                            int sx = Math.Clamp(ix - 1 + kx, 0, src - 1);
                            // +hidden: skip the CLS position; patch grid is row-major sy*src+sx.
                            rowAcc += wx[kx] * src_[hidden + ((long)(sy * src + sx)) * hidden + c];
                        }
                        acc += wy[ky] * rowAcc;
                    }
                    op[dstBase + c] = acc;
                }
            }
        }
        return outp;
    }

    /// <summary>PyTorch cubic-convolution upsample coefficients (A = −0.75) for fractional offset t∈[0,1).</summary>
    private static void CubicCoeffs(float t, Span<float> c)
    {
        const float A = -0.75f;
        c[0] = Conv2(t + 1f, A);
        c[1] = Conv1(t, A);
        c[2] = Conv1(1f - t, A);
        c[3] = Conv2(2f - t, A);
    }

    private static float Conv1(float x, float a) => ((a + 2f) * x - (a + 3f)) * x * x + 1f;
    private static float Conv2(float x, float a) => ((a * x - 5f * a) * x + 8f * a) * x - 4f * a;

    internal static Tensor F32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}

/// <summary>One HF ViT layer: pre-norm self-attention (<c>layernorm_before</c>) + residual, then pre-norm
/// FFN (<c>layernorm_after</c> → <c>intermediate.dense</c> → exact-erf GELU → <c>output.dense</c>) + residual.</summary>
internal sealed unsafe class DinoViTLayer
{
    private readonly int _hidden, _numHeads, _headDim, _inter;
    private readonly float _eps;
    private Tensor? _lnBeforeW, _lnBeforeB, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB;
    private Tensor? _lnAfterW, _lnAfterB, _interW, _interB, _outW, _outB;

    public DinoViTLayer(int hidden, int numHeads, int inter, float eps)
    {
        _hidden = hidden; _numHeads = numHeads; _headDim = hidden / numHeads; _inter = inter; _eps = eps;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _lnBeforeW = DinoViTEncoder.F32(w[$"{prefix}.layernorm_before.weight"]); _lnBeforeB = DinoViTEncoder.F32(w[$"{prefix}.layernorm_before.bias"]);
        _qW = DinoViTEncoder.F32(w[$"{prefix}.attention.attention.query.weight"]); _qB = DinoViTEncoder.F32(w[$"{prefix}.attention.attention.query.bias"]);
        _kW = DinoViTEncoder.F32(w[$"{prefix}.attention.attention.key.weight"]); _kB = DinoViTEncoder.F32(w[$"{prefix}.attention.attention.key.bias"]);
        _vW = DinoViTEncoder.F32(w[$"{prefix}.attention.attention.value.weight"]); _vB = DinoViTEncoder.F32(w[$"{prefix}.attention.attention.value.bias"]);
        _oW = DinoViTEncoder.F32(w[$"{prefix}.attention.output.dense.weight"]); _oB = DinoViTEncoder.F32(w[$"{prefix}.attention.output.dense.bias"]);
        _lnAfterW = DinoViTEncoder.F32(w[$"{prefix}.layernorm_after.weight"]); _lnAfterB = DinoViTEncoder.F32(w[$"{prefix}.layernorm_after.bias"]);
        _interW = DinoViTEncoder.F32(w[$"{prefix}.intermediate.dense.weight"]); _interB = DinoViTEncoder.F32(w[$"{prefix}.intermediate.dense.bias"]);
        _outW = DinoViTEncoder.F32(w[$"{prefix}.output.dense.weight"]); _outB = DinoViTEncoder.F32(w[$"{prefix}.output.dense.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_lnBeforeW, _lnBeforeB, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _lnAfterW, _lnAfterB, _interW, _interB, _outW, _outB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int batch = (int)x.Shape[0], seq = (int)x.Shape[1];
        TensorShape shape = new(batch, seq, _hidden);

        Tensor n1 = new(shape, DType.F32); backend.LayerNorm(n1, x, _lnBeforeW!, _lnBeforeB!, _eps);
        Tensor attn = Attention(backend, n1, batch, seq); n1.Dispose();
        Tensor r1 = new(shape, DType.F32); backend.Add(r1, x, attn); attn.Dispose();

        Tensor n2 = new(shape, DType.F32); backend.LayerNorm(n2, r1, _lnAfterW!, _lnAfterB!, _eps);
        Tensor f1 = new(new TensorShape(batch, seq, _inter), DType.F32); backend.Linear(f1, n2, _interW!, _interB!); n2.Dispose();
        VitOps.ExactGelu((float*)f1.DataPointer, f1.ElementCount);
        Tensor f2 = new(shape, DType.F32); backend.Linear(f2, f1, _outW!, _outB!); f1.Dispose();
        Tensor r2 = new(shape, DType.F32); backend.Add(r2, r1, f2); r1.Dispose(); f2.Dispose();
        return r2;
    }

    private Tensor Attention(IBackend backend, Tensor input, int batch, int seq)
    {
        Tensor q = new(new TensorShape(batch, seq, _hidden), DType.F32);
        Tensor k = new(new TensorShape(batch, seq, _hidden), DType.F32);
        Tensor v = new(new TensorShape(batch, seq, _hidden), DType.F32);
        backend.Linear(q, input, _qW!, _qB!); backend.Linear(k, input, _kW!, _kB!); backend.Linear(v, input, _vW!, _vB!);
        Tensor qh = new(new TensorShape(batch, _numHeads, seq, _headDim), DType.F32); backend.Permute0213(qh, q, seq, _numHeads, _headDim);
        Tensor kh = new(new TensorShape(batch, _numHeads, seq, _headDim), DType.F32); backend.Permute0213(kh, k, seq, _numHeads, _headDim);
        Tensor vh = new(new TensorShape(batch, _numHeads, seq, _headDim), DType.F32); backend.Permute0213(vh, v, seq, _numHeads, _headDim);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor ao = new(new TensorShape(batch, _numHeads, seq, _headDim), DType.F32);
        backend.ScaledDotProductAttention(ao, qh, kh, vh, null, 1f / MathF.Sqrt(_headDim));
        qh.Dispose(); kh.Dispose(); vh.Dispose();
        Tensor merged = new(new TensorShape(batch, seq, _hidden), DType.F32); backend.Permute0213(merged, ao, _numHeads, seq, _headDim); ao.Dispose();
        Tensor proj = new(new TensorShape(batch, seq, _hidden), DType.F32);
        backend.Linear(proj, merged, _oW!, _oB!); merged.Dispose();
        return proj;
    }

    private Tensor ToHeads(Tensor input, int batch, int seq)
    {
        Tensor o = new(new TensorShape(batch, _numHeads, seq, _headDim), DType.F32);
        float* sp = (float*)input.DataPointer; float* dp = (float*)o.DataPointer;
        for (int b = 0; b < batch; b++) for (int s = 0; s < seq; s++) for (int h = 0; h < _numHeads; h++)
            new ReadOnlySpan<float>(sp + (b * seq + s) * _hidden + h * _headDim, _headDim)
                .CopyTo(new Span<float>(dp + ((b * _numHeads + h) * seq + s) * _headDim, _headDim));
        return o;
    }

    private Tensor FromHeads(Tensor input, int batch, int seq)
    {
        Tensor o = new(new TensorShape(batch, seq, _hidden), DType.F32);
        float* sp = (float*)input.DataPointer; float* dp = (float*)o.DataPointer;
        for (int b = 0; b < batch; b++) for (int s = 0; s < seq; s++) for (int h = 0; h < _numHeads; h++)
            new ReadOnlySpan<float>(sp + ((b * _numHeads + h) * seq + s) * _headDim, _headDim)
                .CopyTo(new Span<float>(dp + (b * seq + s) * _hidden + h * _headDim, _headDim));
        return o;
    }
}
