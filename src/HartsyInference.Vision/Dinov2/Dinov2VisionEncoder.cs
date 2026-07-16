using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Dinov2;

/// <summary>DINOv2 ViT backbone. Produces the full token sequence (CLS + register tokens + patch tokens)
/// as <c>[B, seqLen, hidden]</c> — the conditioning that image→3D DiTs cross-attend to. Structure mirrors a
/// standard pre-norm ViT (cf. <c>SiglipVisionEncoder</c>) with two DINOv2-specific additions: a prepended
/// CLS token (+ optional registers) and per-block <b>LayerScale</b> (a learned per-channel γ scaling each
/// sub-layer output before the residual add). All matmuls route through <see cref="IBackend"/>.
/// <para>Position-embedding interpolation (for non-native image sizes) and the giant variant's SwiGLU FFN
/// are not implemented yet — <b>validation-gated</b>; the base/large GELU-MLP variants are the target.</para></summary>
public sealed unsafe class Dinov2VisionEncoder
{
    private readonly Dinov2Preset _preset;
    private readonly Dinov2Layer[] _layers;

    private Tensor? _patchW, _patchB;   // projection.weight [hidden,3,p,p], bias [hidden]
    private Tensor? _clsToken;          // [1,1,hidden]
    private Tensor? _registerTokens;    // [1,regs,hidden] or null
    private Tensor? _posEmb;            // [1, 1+numPatches, hidden]
    private Tensor? _finalLnW, _finalLnB;

    public Dinov2Preset Preset => _preset;
    public int HiddenSize => _preset.HiddenSize;
    public int SequenceLength => _preset.SequenceLength;

    public Dinov2VisionEncoder(Dinov2Preset preset)
    {
        _preset = preset;
        _layers = new Dinov2Layer[preset.NumLayers];
        for (int i = 0; i < preset.NumLayers; i++)
            _layers[i] = new Dinov2Layer(preset.HiddenSize, preset.NumHeads, preset.IntermediateSize, preset.LayerNormEps);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _patchW = F32(weights[$"{p}embeddings.patch_embeddings.projection.weight"]);
        _patchB = F32(weights[$"{p}embeddings.patch_embeddings.projection.bias"]);
        _clsToken = F32(weights[$"{p}embeddings.cls_token"]);
        _posEmb = F32(weights[$"{p}embeddings.position_embeddings"]);
        if (_preset.NumRegisterTokens > 0 && weights.TryGetValue($"{p}embeddings.register_tokens", out Tensor? reg))
            _registerTokens = F32(reg);
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(weights, $"{p}encoder.layer.{i}");
        _finalLnW = F32(weights[$"{p}layernorm.weight"]);
        _finalLnB = F32(weights[$"{p}layernorm.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_patchW, _patchB, _clsToken, _registerTokens, _posEmb, _finalLnW, _finalLnB];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        foreach (Dinov2Layer l in _layers) foreach (Tensor t in l.EnumerateWeights()) yield return t;
    }

    /// <summary>Encodes preprocessed pixels <c>[B,3,imageSize,imageSize]</c> into the token sequence
    /// <c>[B, seqLen, hidden]</c> (CLS + registers + patches). With <paramref name="applyFinalNorm"/> true
    /// (default) the backbone's final layer norm is applied; set false to return dinov2's <c>x_prenorm</c>
    /// (the block output before <c>norm</c>) — what the TRELLIS conditioner taps.</summary>
    public Tensor Encode(IBackend backend, Tensor pixelValues, bool applyFinalNorm = true)
    {
        if (pixelValues.Shape.Rank != 4 || pixelValues.Shape[1] != 3
            || pixelValues.Shape[2] != _preset.ImageSize || pixelValues.Shape[3] != _preset.ImageSize)
            throw new ArgumentException($"pixelValues must be [B,3,{_preset.ImageSize},{_preset.ImageSize}]; got {pixelValues.Shape}.", nameof(pixelValues));

        int batch = (int)pixelValues.Shape[0];
        int hidden = _preset.HiddenSize, grid = _preset.PatchGrid, numPatches = _preset.NumPatches;

        // 1. Patch embed conv → [B,hidden,grid,grid] → [B,numPatches,hidden].
        Tensor patch = new(new TensorShape(batch, hidden, grid, grid), DType.F32);
        backend.Conv2D(patch, pixelValues, _patchW!, _patchB!, _preset.PatchSize, _preset.PatchSize, 0, 0);
        Tensor flat = patch.Reshape(new TensorShape(batch, hidden, numPatches));
        Tensor patches = new(new TensorShape(batch, numPatches, hidden), DType.F32);
        backend.Transpose2D(patches, flat, hidden, numPatches);
        patch.Dispose();

        // 2. Build [CLS, patches], add position embeddings, then insert registers → [CLS, regs, patches].
        Tensor seq = BuildSequence(batch, hidden, numPatches, patches);
        patches.Dispose();

        // 3. Transformer layers.
        Tensor h = seq;
        for (int i = 0; i < _layers.Length; i++) { Tensor n = _layers[i].Forward(backend, h); h.Dispose(); h = n; }

        // 4. Final layer norm (skipped for the x_prenorm tap).
        if (!applyFinalNorm) return h;
        Tensor outp = new(h.Shape, DType.F32);
        backend.LayerNorm(outp, h, _finalLnW!, _finalLnB!, _preset.LayerNormEps);
        h.Dispose();
        return outp;
    }

    private Tensor BuildSequence(int batch, int hidden, int numPatches, Tensor patches)
    {
        int regs = _preset.NumRegisterTokens;
        int seqLen = _preset.SequenceLength;
        Tensor seq = new(new TensorShape(batch, seqLen, hidden), DType.F32);
        float* dp = (float*)seq.DataPointer;
        float* cls = (float*)_clsToken!.DataPointer;     // [1,1,hidden]
        float* pos = (float*)_posEmb!.DataPointer;       // [1, 1+numPatches, hidden]
        float* pat = (float*)patches.DataPointer;        // [B, numPatches, hidden]
        float* reg = _registerTokens is null ? null : (float*)_registerTokens.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            // CLS at position 0: cls_token + pos[0].
            for (int c = 0; c < hidden; c++) dp[(long)(b * seqLen) * hidden + c] = cls[c] + pos[c];
            // Registers (no positional embedding), positions 1..regs.
            for (int r = 0; r < regs; r++)
                for (int c = 0; c < hidden; c++)
                    dp[((long)(b * seqLen) + 1 + r) * hidden + c] = reg![r * hidden + c];
            // Patches at positions 1+regs.., each + pos[1+patchIdx].
            for (int pi = 0; pi < numPatches; pi++)
            {
                long dstBase = ((long)(b * seqLen) + 1 + regs + pi) * hidden;
                long patBase = ((long)b * numPatches + pi) * hidden;
                long posBase = (long)(1 + pi) * hidden;
                for (int c = 0; c < hidden; c++) dp[dstBase + c] = pat[patBase + c] + pos[posBase + c];
            }
        }
        return seq;
    }

    internal static Tensor F32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}

/// <summary>One DINOv2 transformer layer: pre-norm self-attention and MLP, each scaled by a learned
/// per-channel LayerScale γ before its residual add.</summary>
internal sealed unsafe class Dinov2Layer
{
    private readonly int _hidden, _numHeads, _headDim, _inter;
    private readonly float _eps;
    private bool _swiglu;
    private Tensor? _n1W, _n1B, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _ls1;
    private Tensor? _n2W, _n2B, _fc1W, _fc1B, _fc2W, _fc2B, _ls2;
    private Tensor? _swiInW, _swiInB, _swiOutW, _swiOutB;  // SwiGLU FFN (giant variant)

    public Dinov2Layer(int hidden, int numHeads, int inter, float eps)
    {
        _hidden = hidden; _numHeads = numHeads; _headDim = hidden / numHeads; _inter = inter; _eps = eps;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _n1W = Dinov2VisionEncoder.F32(w[$"{prefix}.norm1.weight"]); _n1B = Dinov2VisionEncoder.F32(w[$"{prefix}.norm1.bias"]);
        _qW = Dinov2VisionEncoder.F32(w[$"{prefix}.attention.attention.query.weight"]); _qB = Dinov2VisionEncoder.F32(w[$"{prefix}.attention.attention.query.bias"]);
        _kW = Dinov2VisionEncoder.F32(w[$"{prefix}.attention.attention.key.weight"]); _kB = Dinov2VisionEncoder.F32(w[$"{prefix}.attention.attention.key.bias"]);
        _vW = Dinov2VisionEncoder.F32(w[$"{prefix}.attention.attention.value.weight"]); _vB = Dinov2VisionEncoder.F32(w[$"{prefix}.attention.attention.value.bias"]);
        _oW = Dinov2VisionEncoder.F32(w[$"{prefix}.attention.output.dense.weight"]); _oB = Dinov2VisionEncoder.F32(w[$"{prefix}.attention.output.dense.bias"]);
        // LayerScale is DINOv2-specific; plain DINO/DINOv1 (e.g. the TripoSR tokenizer) omits it → identity.
        _ls1 = w.TryGetValue($"{prefix}.layer_scale1.lambda1", out Tensor? ls1) ? Dinov2VisionEncoder.F32(ls1) : null;
        _n2W = Dinov2VisionEncoder.F32(w[$"{prefix}.norm2.weight"]); _n2B = Dinov2VisionEncoder.F32(w[$"{prefix}.norm2.bias"]);
        // Giant uses a SwiGLU FFN (mlp.weights_in/out); base/large use the gelu fc1/fc2 MLP.
        if (w.TryGetValue($"{prefix}.mlp.weights_in.weight", out Tensor? swiIn))
        {
            _swiglu = true;
            _swiInW = Dinov2VisionEncoder.F32(swiIn); _swiInB = Dinov2VisionEncoder.F32(w[$"{prefix}.mlp.weights_in.bias"]);
            _swiOutW = Dinov2VisionEncoder.F32(w[$"{prefix}.mlp.weights_out.weight"]); _swiOutB = Dinov2VisionEncoder.F32(w[$"{prefix}.mlp.weights_out.bias"]);
        }
        else
        {
            _fc1W = Dinov2VisionEncoder.F32(w[$"{prefix}.mlp.fc1.weight"]); _fc1B = Dinov2VisionEncoder.F32(w[$"{prefix}.mlp.fc1.bias"]);
            _fc2W = Dinov2VisionEncoder.F32(w[$"{prefix}.mlp.fc2.weight"]); _fc2B = Dinov2VisionEncoder.F32(w[$"{prefix}.mlp.fc2.bias"]);
        }
        _ls2 = w.TryGetValue($"{prefix}.layer_scale2.lambda1", out Tensor? ls2) ? Dinov2VisionEncoder.F32(ls2) : null;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_n1W, _n1B, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _ls1, _n2W, _n2B, _fc1W, _fc1B, _fc2W, _fc2B, _ls2, _swiInW, _swiInB, _swiOutW, _swiOutB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int batch = (int)x.Shape[0], seq = (int)x.Shape[1];
        TensorShape shape = new(batch, seq, _hidden);

        Tensor n1 = new(shape, DType.F32); backend.LayerNorm(n1, x, _n1W!, _n1B!, _eps);
        Tensor attn = LayerScale(backend, Attention(backend, n1, batch, seq), _ls1, batch, seq); n1.Dispose();
        Tensor r1 = new(shape, DType.F32); backend.Add(r1, x, attn); attn.Dispose();

        Tensor n2 = new(shape, DType.F32); backend.LayerNorm(n2, r1, _n2W!, _n2B!, _eps);
        Tensor mlp = LayerScale(backend, Mlp(backend, n2, batch, seq), _ls2, batch, seq); n2.Dispose();
        Tensor r2 = new(shape, DType.F32); backend.Add(r2, r1, mlp); r1.Dispose(); mlp.Dispose();
        return r2;
    }

    // GPU-resident LayerScale: t[b,s,c] *= gamma[c]. Was a host DataPointer loop that drained the compute stream
    // every block (2×/block × NumLayers → the DINOv2-giant conditioner's dominant cost). AffineBroadcastLastDim
    // (scale-only, shift=null) broadcasts gamma[hidden] over the (b,s) rows on-device. Batch>1 keeps the host
    // path (the affine kernel indexes scale[b·dim+d], valid for the B=1 conditioning use here).
    private Tensor LayerScale(IBackend backend, Tensor t, Tensor? gamma, int batch, int seq)
    {
        if (gamma is null) return t; // no LayerScale (plain DINO) → identity
        if (batch == 1)
        {
            Tensor o = new(t.Shape, DType.F32);
            backend.AffineBroadcastLastDim(o, t, gamma, null);
            t.Dispose();
            return o;
        }
        float* tp = (float*)t.DataPointer; float* g = (float*)gamma.DataPointer;
        for (long i = 0; i < (long)batch * seq; i++)
            for (int c = 0; c < _hidden; c++) tp[i * _hidden + c] *= g[c];
        return t;
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

    private Tensor Mlp(IBackend backend, Tensor input, int batch, int seq)
    {
        if (_swiglu)
        {
            // weights_in -> [.,2H], split (x1,x2); silu(x1)*x2; weights_out.
            // GPU-resident SwiGLU: proj[.,2H] → split (x1,x2) → silu(x1)·x2 → weights_out. The old host silu-gate
            // loop read proj.DataPointer, draining the compute stream every block (the sync-h2d-stream-drain disease).
            int twoH = (int)_swiInW!.Shape[0], h2 = twoH / 2;
            TensorShape half = new(batch, seq, h2);
            Tensor proj = new(new TensorShape(batch, seq, twoH), DType.F32); backend.Linear(proj, input, _swiInW!, _swiInB!);
            Tensor x1 = new(half, DType.F32); backend.SliceLastDim(x1, proj, 0);
            Tensor x2 = new(half, DType.F32); backend.SliceLastDim(x2, proj, h2); proj.Dispose();
            Tensor sil = new(half, DType.F32); backend.Silu(sil, x1); x1.Dispose();
            Tensor gated = new(half, DType.F32); backend.Mul(gated, sil, x2); sil.Dispose(); x2.Dispose();
            Tensor outp = new(new TensorShape(batch, seq, _hidden), DType.F32); backend.Linear(outp, gated, _swiOutW!, _swiOutB!); gated.Dispose();
            return outp;
        }
        Tensor f1 = new(new TensorShape(batch, seq, _inter), DType.F32); backend.Linear(f1, input, _fc1W!, _fc1B!);
        Tensor act = new(f1.Shape, DType.F32); backend.Gelu(act, f1); f1.Dispose();
        Tensor f2 = new(new TensorShape(batch, seq, _hidden), DType.F32); backend.Linear(f2, act, _fc2W!, _fc2B!); act.Dispose();
        return f2;
    }
}
