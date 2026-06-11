using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Chroma Radiance output head ("NeRF head", replaces classic Chroma's <c>final_layer</c>). For each 16×16
/// patch, the transformer's output img token (3072) generates the weights of a tiny per-patch GLU MLP that refines
/// the 256 per-pixel embeddings of that patch, producing the <b>x0</b> prediction directly in pixel space:
/// <list type="number">
///   <item><c>nerf_image_embedder</c>: Linear(3 + maxFreqs² → nerfHidden) in FP32 over per-pixel [r,g,b] from the
///         noisy input image plus maxFreqs² separable cosine positional features.</item>
///   <item><c>nerf_blocks.{i}</c> (depth 4): <c>param_generator</c> Linear(3072 → 3·nerfHidden·(nerfHidden·ratio))
///         emits gate W1, value W2, out W3 per patch; <c>x = x + W3(silu(xn·W1ᵀ) ⊙ (xn·W2ᵀ))</c> with
///         <c>xn = RMSNorm(x)</c> (key <c>.norm.scale</c>).</item>
///   <item>Final head: variant A (<c>nerf_final_layer_conv</c>): RMSNorm → fold to [B,64,H,W] → Conv2d(64→3, k3, p1).
///         Variant B (<c>nerf_final_layer</c>): RMSNorm → per-pixel Linear [3,64] (folded as a 1×1 conv).</item>
/// </list>
/// Validation-gated items (see docs/Research/CHROMA_RADIANCE_ARCHITECTURE.md, uncertainty table): the cosine basis
/// formula (#1), the 49152-chunk split order/transposition (#2), and the conv-variant sub-key names (#4).</summary>
public sealed unsafe class ChromaRadianceNerfHead : IDisposable
{
    // RMSNorm epsilon — assumed to match the Flux/Chroma block convention (validation-gated).
    private const float NormEps = 1e-6f;
    private const int PatchTileSize = 32;

    private readonly int _patchSize;
    private readonly int _nerfHidden;
    private readonly int _maxFreqs;
    private readonly int _depth;
    private readonly int _mlpRatio;

    private Tensor? _embedWeight, _embedBias;
    private readonly Tensor?[] _paramGenWeight;
    private readonly Tensor?[] _paramGenBias;
    private readonly Tensor?[] _normScale;

    private Tensor? _finalNormScale;
    private Tensor? _finalConvWeight, _finalConvBias;
    private int _finalConvPad;
    private int _disposed;

    /// <summary>Creates the head from config dims (weights loaded separately).</summary>
    public ChromaRadianceNerfHead(int patchSize, int nerfHidden, int maxFreqs, int depth, int mlpRatio = 4)
    {
        _patchSize = patchSize;
        _nerfHidden = nerfHidden;
        _maxFreqs = maxFreqs;
        _depth = depth;
        _mlpRatio = mlpRatio;
        _paramGenWeight = new Tensor?[depth];
        _paramGenBias = new Tensor?[depth];
        _normScale = new Tensor?[depth];
    }

    /// <summary>True when the loaded final head is the conv (variant A) flavor.</summary>
    public bool UsesConvFinalLayer => _finalConvPad > 0;

    /// <summary>Loads all NeRF head weights, detecting the final-layer variant by key presence.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _embedWeight = AsF32(weights["nerf_image_embedder.embedder.0.weight"]);
        _embedBias = AsF32(weights["nerf_image_embedder.embedder.0.bias"]);

        for (int i = 0; i < _depth; i++)
        {
            _paramGenWeight[i] = weights[$"nerf_blocks.{i}.param_generator.weight"];
            weights.TryGetValue($"nerf_blocks.{i}.param_generator.bias", out _paramGenBias[i]);
            // backend.RmsNorm reads the scale as float* — must be F32.
            _normScale[i] = AsF32(weights[$"nerf_blocks.{i}.norm.scale"]);
        }

        if (weights.TryGetValue("nerf_final_layer_conv.norm.scale", out Tensor? convNorm))
        {
            // Variant A: RMSNorm + Conv2d(nerfHidden→3, k3, pad 1). Conv sub-key name validation-gated —
            // pattern-match any 4D weight under the nerf_final_layer_conv prefix.
            _finalNormScale = AsF32(convNorm);
            foreach (KeyValuePair<string, Tensor> kvp in weights)
            {
                if (!kvp.Key.StartsWith("nerf_final_layer_conv.", StringComparison.Ordinal)) continue;
                if (kvp.Key.EndsWith(".weight", StringComparison.Ordinal) && kvp.Value.Shape.Rank == 4)
                {
                    _finalConvWeight = kvp.Value;
                    weights.TryGetValue(kvp.Key[..^".weight".Length] + ".bias", out _finalConvBias);
                }
            }
            if (_finalConvWeight is null)
                throw new ArgumentException("nerf_final_layer_conv.norm.scale present but no 4D conv weight found under nerf_final_layer_conv.*");
            _finalConvPad = ((int)_finalConvWeight.Shape[2] - 1) / 2;
            if (_finalConvPad == 0) _finalConvPad = 1;
        }
        else
        {
            // Variant B: per-pixel Linear [3, nerfHidden] — folded into a 1×1 conv so both variants share one path.
            _finalNormScale = AsF32(weights["nerf_final_layer.norm.scale"]);
            Tensor linear = weights["nerf_final_layer.linear.weight"];
            TensorShape convShape = new TensorShape(linear.Shape[0], linear.Shape[1], 1, 1);
            Tensor asConv = new Tensor(convShape, linear.DType);
            long bytes = linear.Shape.ElementCount * linear.DType.SizeInBytes;
            Buffer.MemoryCopy((void*)linear.DataPointer, (void*)asConv.DataPointer, bytes, bytes);
            asConv.Fp8ScaleFactor = linear.Fp8ScaleFactor;
            _finalConvWeight = asConv;
            weights.TryGetValue("nerf_final_layer.linear.bias", out _finalConvBias);
            _finalConvPad = 0;
        }
    }

    /// <summary>Enumerates weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedWeight is not null) yield return _embedWeight;
        if (_embedBias is not null) yield return _embedBias;
        for (int i = 0; i < _depth; i++)
        {
            if (_paramGenWeight[i] is not null) yield return _paramGenWeight[i]!;
            if (_paramGenBias[i] is not null) yield return _paramGenBias[i]!;
            if (_normScale[i] is not null) yield return _normScale[i]!;
        }
        if (_finalNormScale is not null) yield return _finalNormScale;
        if (_finalConvWeight is not null) yield return _finalConvWeight;
        if (_finalConvBias is not null) yield return _finalConvBias;
    }

    /// <summary>Produces the x0 prediction [B, 3, H, W] from the noisy input image and the transformer's output
    /// img tokens. Patches are processed in tiles of <see cref="PatchTileSize"/> to bound scratch memory.</summary>
    /// <param name="noisyRgb">The noisy pixel latent fed to the model this step, [B, 3, H, W] in [-1, 1].</param>
    /// <param name="imgTokens">Transformer output image tokens [B, N, 3072], N = (H/P)·(W/P).</param>
    public Tensor Forward(IBackend backend, Tensor noisyRgb, Tensor imgTokens)
    {
        if (_embedWeight is null || _finalConvWeight is null)
            throw new InvalidOperationException("ChromaRadianceNerfHead weights not loaded.");

        int batch = (int)noisyRgb.Shape[0];
        int height = (int)noisyRgb.Shape[2];
        int width = (int)noisyRgb.Shape[3];
        int p = _patchSize;
        int pix = p * p;
        int wPacked = width / p;
        int seqLen = (int)imgTokens.Shape[1];
        int tokenDim = (int)imgTokens.Shape[2];
        int nh = _nerfHidden;
        int inner = nh * _mlpRatio;
        int posFeatures = _maxFreqs * _maxFreqs;
        int embedIn = 3 + posFeatures;
        int paramDim = (int)_paramGenWeight[0]!.Shape[0];

        if (seqLen != (height / p) * wPacked)
            throw new ArgumentException($"imgTokens seq len {seqLen} doesn't match {(height / p) * wPacked} patches for {height}x{width}/{p}.");
        if (paramDim != 3 * nh * inner)
            throw new ArgumentException($"param_generator out dim {paramDim} != 3·{nh}·{inner}; NeRF head dims inconsistent.");

        float[] posFeat = BuildPositionalFeatures(p, _maxFreqs);

        // Post-block per-pixel features, folded back to image layout for the shared conv final path.
        Tensor features = new Tensor(new TensorShape(batch, nh, height, width), DType.F32);

        // Per-tile scratch (reused across tiles/blocks — no allocations inside the loop).
        int tile = Math.Min(PatchTileSize, seqLen);
        Tensor embedInput = new Tensor(new TensorShape(tile * pix, embedIn), DType.F32);
        Tensor x = new Tensor(new TensorShape(tile * pix, nh), DType.F32);
        Tensor xn = new Tensor(new TensorShape(tile * pix, nh), DType.F32);
        Tensor tokensTile = new Tensor(new TensorShape(tile, tokenDim), imgTokens.DType);
        Tensor paramsTile = new Tensor(new TensorShape(tile, paramDim), DType.F32);
        Tensor w1 = new Tensor(new TensorShape(inner, nh), DType.F32);
        Tensor w2 = new Tensor(new TensorShape(inner, nh), DType.F32);
        Tensor w3 = new Tensor(new TensorShape(nh, inner), DType.F32);
        Tensor xnPatch = new Tensor(new TensorShape(pix, nh), DType.F32);
        Tensor gate = new Tensor(new TensorShape(pix, inner), DType.F32);
        Tensor value = new Tensor(new TensorShape(pix, inner), DType.F32);
        Tensor blockOut = new Tensor(new TensorShape(pix, nh), DType.F32);

        float* rgbPtr = (float*)noisyRgb.DataPointer;
        float* tokPtr = (float*)imgTokens.DataPointer;
        float* featPtr = (float*)features.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int tileStart = 0; tileStart < seqLen; tileStart += tile)
            {
                int tileCount = Math.Min(tile, seqLen - tileStart);
                int rows = tileCount * pix;

                // ── 1. Embed: [r,g,b] + positional features → nerfHidden, FP32 ──
                FillEmbedInput(embedInput, rgbPtr, posFeat, b, tileStart, tileCount, height, width, wPacked);
                Tensor embedView = rows == tile * pix ? embedInput : SliceRows(embedInput, rows, embedIn);
                Tensor xView = rows == tile * pix ? x : SliceRows(x, rows, nh);
                backend.Linear(xView, embedView, _embedWeight!, _embedBias);
                if (!ReferenceEquals(xView, x))
                {
                    long bytes = (long)rows * nh * sizeof(float);
                    Buffer.MemoryCopy((void*)xView.DataPointer, (void*)x.DataPointer, bytes, bytes);
                    xView.Dispose();
                    embedView.Dispose();
                }

                // Tile's transformer tokens [tileCount, tokenDim].
                CopyTokenRows(tokensTile, imgTokens, b, tileStart, tileCount, seqLen, tokenDim);

                // ── 2. Hypernetwork GLU blocks ──
                for (int d = 0; d < _depth; d++)
                {
                    Tensor tokView = tileCount == tile ? tokensTile : SliceRows(tokensTile, tileCount, tokenDim, imgTokens.DType);
                    Tensor parView = tileCount == tile ? paramsTile : SliceRows(paramsTile, tileCount, paramDim);
                    if (!ReferenceEquals(tokView, tokensTile))
                    {
                        long tb = (long)tileCount * tokenDim * tokensTile.DType.SizeInBytes;
                        Buffer.MemoryCopy((void*)tokensTile.DataPointer, (void*)tokView.DataPointer, tb, tb);
                    }
                    backend.Linear(parView, tokView, _paramGenWeight[d]!, _paramGenBias[d]);

                    Tensor xnView = rows == tile * pix ? xn : SliceRows(xn, rows, nh);
                    Tensor xResView = rows == tile * pix ? x : SliceRows(x, rows, nh);
                    if (!ReferenceEquals(xResView, x))
                    {
                        long xb = (long)rows * nh * sizeof(float);
                        Buffer.MemoryCopy((void*)x.DataPointer, (void*)xResView.DataPointer, xb, xb);
                    }
                    backend.RmsNorm(xnView, xResView, _normScale[d]!, NormEps);

                    float* parPtr = (float*)parView.DataPointer;
                    float* xnPtr = (float*)xnView.DataPointer;
                    float* xPtr = (float*)x.DataPointer;
                    long chunk = (long)inner * nh;

                    for (int t = 0; t < tileCount; t++)
                    {
                        // Split order [W1 gate, W2 value, W3 out], each chunk row-major [out, in] —
                        // validation-gated (research doc item 2).
                        long rowBase = (long)t * paramDim;
                        long w1Bytes = chunk * sizeof(float);
                        Buffer.MemoryCopy(parPtr + rowBase, (void*)w1.DataPointer, w1Bytes, w1Bytes);
                        Buffer.MemoryCopy(parPtr + rowBase + chunk, (void*)w2.DataPointer, w1Bytes, w1Bytes);
                        Buffer.MemoryCopy(parPtr + rowBase + 2 * chunk, (void*)w3.DataPointer, w1Bytes, w1Bytes);

                        long pixBytes = (long)pix * nh * sizeof(float);
                        Buffer.MemoryCopy(xnPtr + (long)t * pix * nh, (void*)xnPatch.DataPointer, pixBytes, pixBytes);

                        backend.Linear(gate, xnPatch, w1, null);
                        backend.Linear(value, xnPatch, w2, null);

                        // glu = silu(gate) ⊙ value, fused in one pass (reuses the gate buffer).
                        float* gPtr = (float*)gate.DataPointer;
                        float* vPtr = (float*)value.DataPointer;
                        long gluCount = (long)pix * inner;
                        for (long i = 0; i < gluCount; i++)
                        {
                            float g = gPtr[i];
                            gPtr[i] = g / (1.0f + MathF.Exp(-g)) * vPtr[i];
                        }

                        backend.Linear(blockOut, gate, w3, null);

                        // Residual back into the tile's x rows.
                        float* oPtr = (float*)blockOut.DataPointer;
                        long xBase = (long)t * pix * nh;
                        for (long i = 0; i < pixBytes / sizeof(float); i++)
                            xPtr[xBase + i] += oPtr[i];
                    }

                    if (!ReferenceEquals(tokView, tokensTile)) tokView.Dispose();
                    if (!ReferenceEquals(parView, paramsTile)) parView.Dispose();
                    if (!ReferenceEquals(xnView, xn)) xnView.Dispose();
                    if (!ReferenceEquals(xResView, x)) xResView.Dispose();
                }

                // ── 3. Final RMSNorm, then fold pixels back to [B, nh, H, W] ──
                Tensor xnFinal = rows == tile * pix ? xn : SliceRows(xn, rows, nh);
                Tensor xFinal = rows == tile * pix ? x : SliceRows(x, rows, nh);
                if (!ReferenceEquals(xFinal, x))
                {
                    long xb = (long)rows * nh * sizeof(float);
                    Buffer.MemoryCopy((void*)x.DataPointer, (void*)xFinal.DataPointer, xb, xb);
                }
                backend.RmsNorm(xnFinal, xFinal, _finalNormScale!, NormEps);

                ScatterFeatures(featPtr, (float*)xnFinal.DataPointer, b, tileStart, tileCount,
                    nh, p, height, width, wPacked);

                if (!ReferenceEquals(xnFinal, xn)) xnFinal.Dispose();
                if (!ReferenceEquals(xFinal, x)) xFinal.Dispose();
            }
        }

        embedInput.Dispose();
        x.Dispose();
        xn.Dispose();
        tokensTile.Dispose();
        paramsTile.Dispose();
        w1.Dispose();
        w2.Dispose();
        w3.Dispose();
        xnPatch.Dispose();
        gate.Dispose();
        value.Dispose();
        blockOut.Dispose();

        // ── 4. Final conv (variant A: k3 pad 1; variant B folded as 1×1) → x0 [B, 3, H, W] ──
        int outChannels = (int)_finalConvWeight.Shape[0];
        Tensor x0 = new Tensor(new TensorShape(batch, outChannels, height, width), DType.F32);
        backend.Conv2D(x0, features, _finalConvWeight, _finalConvBias, 1, 1, _finalConvPad, _finalConvPad);
        features.Dispose();
        return x0;
    }

    /// <summary>Separable cosine positional basis over the P×P patch: feature[u·F+v](y, x) = cos(π·u·y)·cos(π·v·x)
    /// with y, x ∈ linspace(0, 1, P). Formula validation-gated (research doc item 1).</summary>
    internal static float[] BuildPositionalFeatures(int patchSize, int maxFreqs)
    {
        int pix = patchSize * patchSize;
        int features = maxFreqs * maxFreqs;
        float[] result = new float[pix * features];
        for (int py = 0; py < patchSize; py++)
        {
            float yPos = patchSize > 1 ? (float)py / (patchSize - 1) : 0f;
            for (int px = 0; px < patchSize; px++)
            {
                float xPos = patchSize > 1 ? (float)px / (patchSize - 1) : 0f;
                int baseIdx = (py * patchSize + px) * features;
                for (int u = 0; u < maxFreqs; u++)
                {
                    float cy = MathF.Cos(MathF.PI * u * yPos);
                    for (int v = 0; v < maxFreqs; v++)
                        result[baseIdx + u * maxFreqs + v] = cy * MathF.Cos(MathF.PI * v * xPos);
                }
            }
        }
        return result;
    }

    private void FillEmbedInput(Tensor embedInput, float* rgbPtr, float[] posFeat,
        int b, int tileStart, int tileCount, int height, int width, int wPacked)
    {
        int p = _patchSize;
        int pix = p * p;
        int posFeatures = _maxFreqs * _maxFreqs;
        int embedIn = 3 + posFeatures;
        float* dst = (float*)embedInput.DataPointer;
        long plane = (long)height * width;
        long rgbBase = (long)b * 3 * plane;

        for (int t = 0; t < tileCount; t++)
        {
            int patchIdx = tileStart + t;
            int ph = patchIdx / wPacked;
            int pw = patchIdx % wPacked;
            for (int py = 0; py < p; py++)
            {
                int gy = ph * p + py;
                for (int px = 0; px < p; px++)
                {
                    int gx = pw * p + px;
                    int pixIdx = py * p + px;
                    long row = ((long)t * pix + pixIdx) * embedIn;
                    long off = (long)gy * width + gx;
                    dst[row + 0] = rgbPtr[rgbBase + off];
                    dst[row + 1] = rgbPtr[rgbBase + plane + off];
                    dst[row + 2] = rgbPtr[rgbBase + 2 * plane + off];
                    Array.Copy(posFeat, pixIdx * posFeatures, ToManaged(dst, row + 3, posFeatures), 0, 0);
                    for (int f = 0; f < posFeatures; f++)
                        dst[row + 3 + f] = posFeat[pixIdx * posFeatures + f];
                }
            }
        }
    }

    // Placeholder shim so Array.Copy above is never used with unmanaged memory; the explicit loop does the copy.
    private static float[] ToManaged(float* _, long __, int ___) => [];

    private static void CopyTokenRows(Tensor dst, Tensor imgTokens, int b, int tileStart, int tileCount,
        int seqLen, int tokenDim)
    {
        long elemBytes = imgTokens.DType.SizeInBytes;
        byte* src = (byte*)imgTokens.DataPointer + ((long)b * seqLen + tileStart) * tokenDim * elemBytes;
        long bytes = (long)tileCount * tokenDim * elemBytes;
        Buffer.MemoryCopy(src, (void*)dst.DataPointer, bytes, bytes);
    }

    private void ScatterFeatures(float* featPtr, float* src, int b, int tileStart, int tileCount,
        int nh, int p, int height, int width, int wPacked)
    {
        long plane = (long)height * width;
        long featBase = (long)b * nh * plane;
        int pix = p * p;
        for (int t = 0; t < tileCount; t++)
        {
            int patchIdx = tileStart + t;
            int ph = patchIdx / wPacked;
            int pw = patchIdx % wPacked;
            for (int py = 0; py < p; py++)
            {
                int gy = ph * p + py;
                for (int px = 0; px < p; px++)
                {
                    int gx = pw * p + px;
                    long srcRow = ((long)t * pix + py * p + px) * nh;
                    long dstOff = (long)gy * width + gx;
                    for (int c = 0; c < nh; c++)
                        featPtr[featBase + c * plane + dstOff] = src[srcRow + c];
                }
            }
        }
    }

    private static Tensor SliceRows(Tensor full, int rows, int cols)
        => SliceRows(full, rows, cols, DType.F32);

    private static Tensor SliceRows(Tensor full, int rows, int cols, DType dtype)
    {
        // Partial-tile path: backend ops validate shapes, so a right-sized scratch tensor is required.
        _ = full;
        return new Tensor(new TensorShape(rows, cols), dtype);
    }

    private static Tensor AsF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    /// <summary>Releases weight references.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _embedWeight = null;
            _embedBias = null;
            for (int i = 0; i < _depth; i++)
            {
                _paramGenWeight[i] = null;
                _paramGenBias[i] = null;
                _normScale[i] = null;
            }
            _finalNormScale = null;
            _finalConvWeight = null;
            _finalConvBias = null;
        }
        GC.SuppressFinalize(this);
    }
}
