using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Chroma Radiance output head ("NeRF head", replaces classic Chroma's <c>final_layer</c>). For each 16×16
/// patch, the transformer's output img token (3072) generates the weights of a tiny per-patch GLU MLP that refines
/// the 256 per-pixel embeddings of that patch, producing the <b>x0</b> prediction directly in pixel space:
/// <list type="number">
///   <item><c>nerf_image_embedder</c>: Linear(3 + maxFreqs² → nerfHidden) in FP32 over per-pixel [r,g,b] from the
///         noisy input image plus maxFreqs² separable cosine positional features. Folded at load time into a single
///         strided Conv2d (see <see cref="EmbedPixels"/>) so the embed runs as one device op.</item>
///   <item><c>nerf_blocks.{i}</c> (depth 4): <c>param_generator</c> Linear(3072 → 3·nerfHidden·(nerfHidden·ratio))
///         emits gate W1, value W2, out W3 per patch (each L2-normalized along the input dim);
///         <c>x = x + (silu(xn·W1) ⊙ (xn·W2))·W3</c> with <c>xn = RMSNorm(x)</c> (key <c>.norm.scale</c>).
///         Runs as batched device ops: one param_generator GEMM over all patches, chunk slice + batched-transpose +
///         RMSNorm-as-L2-normalize per chunk, then per-patch <c>BatchedMatMul</c> (batch = patches).</item>
///   <item>Final head: variant A (<c>nerf_final_layer_conv</c>): RMSNorm → fold to [B,64,H,W] → Conv2d(64→3, k3, p1).
///         Variant B (<c>nerf_final_layer</c>): RMSNorm → per-pixel Linear [3,64] (folded as a 1×1 conv).</item>
/// </list>
/// The cosine basis (#1) and param-chunk layout/normalization (#2) are verified against ComfyUI
/// <c>comfy/ldm/chroma_radiance/layers.py</c> (<c>NerfEmbedder.fetch_pos</c> / <c>NerfGLUBlock.forward</c>);
/// the conv-variant sub-key names (#4) remain checkpoint-validation-gated. See
/// docs/Research/CHROMA_RADIANCE_ARCHITECTURE.md, uncertainty table.
/// <para>The former implementation processed patches in host-side tiles — ~50k tiny <c>backend.Linear</c> calls plus
/// host normalize/silu/scatter loops per forward (~35 s/step at 1024²). This device port keeps every stage on the
/// GPU via existing backend ops; no DataPointer is read anywhere in the forward.</para></summary>
public sealed unsafe class ChromaRadianceNerfHead : IDisposable
{
    // RMSNorm epsilon — assumed to match the Flux/Chroma block convention (validation-gated).
    private const float NormEps = 1e-6f;
    // Effectively-zero epsilon for the RMSNorm-based L2 normalize (F.normalize clamps ||x|| at 1e-12).
    private const float L2Eps = 1e-20f;
    // Patches per device tile inside the GLU blocks. Bounds the transient footprint (gate/value/glu are
    // [tile, P², inner] ≈ 268 MB at 1024 patches) so the head fits beside the ~19 GB resident transformer
    // without async-pool OOM-retry churn. Same math, just chunked GEMMs — device-exact.
    private const int PatchTileSize = 1024;

    private readonly int _patchSize;
    private readonly int _nerfHidden;
    private readonly int _maxFreqs;
    private readonly int _depth;
    private readonly int _mlpRatio;

    private Tensor? _embedWeight, _embedBias;
    private Tensor? _embedConvWeight, _embedConvBias;
    private readonly Tensor?[] _paramGenWeight;
    private readonly Tensor?[] _paramGenBias;
    private readonly Tensor?[] _normScale;

    // Constant RMSNorm scale vectors that turn backend.RmsNorm into an exact L2 normalize:
    // x / sqrt(mean(x²)) · (1/sqrt(dim)) == x / sqrt(Σx²).
    private Tensor? _l2ScaleNh;
    private Tensor? _l2ScaleInner;

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

    /// <summary>Loads all NeRF head weights, detecting the final-layer variant by key presence. Also builds the
    /// folded embed conv and the constant L2-normalize scale vectors consumed by the device forward.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _embedWeight = AsF32(weights["nerf_image_embedder.embedder.0.weight"]);
        _embedBias = AsF32(weights["nerf_image_embedder.embedder.0.bias"]);
        BuildFoldedEmbedConv();
        BuildL2Scales();

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
        if (_embedConvWeight is not null) yield return _embedConvWeight;
        if (_embedConvBias is not null) yield return _embedConvBias;
        if (_l2ScaleNh is not null) yield return _l2ScaleNh;
        if (_l2ScaleInner is not null) yield return _l2ScaleInner;
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
    /// img tokens — <see cref="EmbedPixels"/> + <see cref="ForwardFromEmbed"/> in one call.</summary>
    /// <param name="noisyRgb">The noisy pixel latent fed to the model this step, [B, 3, H, W] in [-1, 1].</param>
    /// <param name="imgTokens">Transformer output image tokens [B, N, 3072], N = (H/P)·(W/P).</param>
    public Tensor Forward(IBackend backend, Tensor noisyRgb, Tensor imgTokens)
    {
        Tensor embed = EmbedPixels(backend, noisyRgb);
        Tensor x0 = ForwardFromEmbed(backend, embed, imgTokens, noisyRgb.Shape);
        embed.Dispose();
        return x0;
    }

    /// <summary>Per-pixel embedding [B·N, P², nerfHidden] for the noisy image via the folded embed conv: the
    /// Linear's RGB columns become a stride-P conv and the constant positional-feature contribution is pre-summed
    /// into the per-output-channel bias. Step-invariant across the CFG pair — compute once, share both passes.</summary>
    public Tensor EmbedPixels(IBackend backend, Tensor noisyRgb)
    {
        if (_embedConvWeight is null)
            throw new InvalidOperationException("ChromaRadianceNerfHead weights not loaded.");

        int batch = (int)noisyRgb.Shape[0];
        int height = (int)noisyRgb.Shape[2];
        int width = (int)noisyRgb.Shape[3];
        int p = _patchSize;
        int pix = p * p;
        int seqLen = (height / p) * (width / p);

        Tensor conv = new Tensor(new TensorShape(batch, (long)pix * _nerfHidden, height / p, width / p), DType.F32);
        backend.Conv2D(conv, noisyRgb, _embedConvWeight, _embedConvBias, p, p, 0, 0);

        // [B, pix·nh, N] → [B, N, pix·nh] ≡ [B·N, pix, nh]: batched channel/seq flip.
        Tensor embed = new Tensor(new TensorShape((long)batch * seqLen, pix, _nerfHidden), DType.F32);
        backend.Permute0213(embed, conv, pix * _nerfHidden, seqLen, 1);
        conv.Dispose();
        return embed;
    }

    /// <summary>Hypernetwork GLU blocks + final norm/conv from a precomputed pixel embedding (not consumed — the
    /// CFG pair shares it). All stages are device ops batched over patches.</summary>
    /// <param name="embed">Pixel embedding from <see cref="EmbedPixels"/>, [B·N, P², nerfHidden].</param>
    /// <param name="imgTokens">Transformer output image tokens [B, N, 3072].</param>
    /// <param name="imageShape">The [B, 3, H, W] shape of the noisy input (defines the output geometry).</param>
    public Tensor ForwardFromEmbed(IBackend backend, Tensor embed, Tensor imgTokens, TensorShape imageShape)
    {
        if (_embedConvWeight is null || _finalConvWeight is null)
            throw new InvalidOperationException("ChromaRadianceNerfHead weights not loaded.");

        int batch = (int)imageShape[0];
        int height = (int)imageShape[2];
        int width = (int)imageShape[3];
        int p = _patchSize;
        int pix = p * p;
        int hPacked = height / p;
        int wPacked = width / p;
        int seqLen = (int)imgTokens.Shape[1];
        int nh = _nerfHidden;
        int inner = nh * _mlpRatio;
        long bn = (long)batch * seqLen;
        long chunk = (long)nh * inner;
        int paramDim = (int)_paramGenWeight[0]!.Shape[0];

        if (seqLen != hPacked * wPacked)
            throw new ArgumentException($"imgTokens seq len {seqLen} doesn't match {hPacked * wPacked} patches for {height}x{width}/{p}.");
        if (paramDim != 3 * chunk)
            throw new ArgumentException($"param_generator out dim {paramDim} != 3·{nh}·{inner}; NeRF head dims inconsistent.");

        Tensor x = embed;
        int tile = (int)Math.Min(PatchTileSize, bn);
        int tileCount = (int)((bn + tile - 1) / tile);
        for (int d = 0; d < _depth; d++)
        {
            // ── x = x + (silu(xn·W1) ⊙ (xn·W2))·W3, xn = RMSNorm(x) — batched over patches, in patch tiles. ──
            Tensor xn = new Tensor(x.Shape, DType.F32);
            backend.RmsNorm(xn, x, _normScale[d]!, NormEps);

            Tensor[] outTiles = new Tensor[tileCount];
            for (int t = 0; t < tileCount; t++)
            {
                int start = t * tile;
                int count = (int)Math.Min(tile, bn - start);

                // ── Per-patch generated weights: one GEMM over the tile, then chunk split [W1, W2, W3]. ──
                Tensor tokensTile;
                if (tileCount == 1)
                {
                    tokensTile = imgTokens;
                }
                else
                {
                    tokensTile = new Tensor(new TensorShape(count, imgTokens.Shape[imgTokens.Shape.Rank - 1]), DType.F32);
                    backend.SliceRows(tokensTile, imgTokens, start);
                }
                Tensor genParams = new Tensor(new TensorShape(count, paramDim), DType.F32);
                backend.Linear(genParams, tokensTile, _paramGenWeight[d]!, _paramGenBias[d]);
                if (!ReferenceEquals(tokensTile, imgTokens)) tokensTile.Dispose();

                // Chunks are viewed row-major [inDim, outDim] per patch (fc1/fc2: [nh, inner], fc3: [inner, nh])
                // and L2-normalized along the INPUT dim per output unit (F.normalize(dim=-2)). Batched-transpose
                // to put the normalize dim last, RMSNorm with the constant 1/sqrt(dim) scale (== exact L2
                // normalize), then transpose back to the [in, out] layout BatchedMatMul consumes directly.
                Tensor w1 = NormalizedChunk(backend, genParams, offset: 0, inDim: nh, outDim: inner, count, _l2ScaleNh!);
                Tensor w2 = NormalizedChunk(backend, genParams, offset: (int)chunk, inDim: nh, outDim: inner, count, _l2ScaleNh!);
                Tensor w3 = NormalizedChunk(backend, genParams, offset: (int)(2 * chunk), inDim: inner, outDim: nh, count, _l2ScaleInner!);
                genParams.Dispose();

                Tensor xnTile;
                if (tileCount == 1)
                {
                    xnTile = xn;
                }
                else
                {
                    xnTile = new Tensor(new TensorShape(count, pix, nh), DType.F32);
                    backend.SliceRows(xnTile, xn, start * pix);
                }
                Tensor gate = new Tensor(new TensorShape(count, pix, inner), DType.F32);
                backend.BatchedMatMul(gate, xnTile, w1);
                Tensor value = new Tensor(new TensorShape(count, pix, inner), DType.F32);
                backend.BatchedMatMul(value, xnTile, w2);
                if (!ReferenceEquals(xnTile, xn)) xnTile.Dispose();
                w1.Dispose();
                w2.Dispose();

                Tensor gateSilu = new Tensor(gate.Shape, DType.F32);
                backend.Silu(gateSilu, gate);
                gate.Dispose();
                Tensor glu = new Tensor(gateSilu.Shape, DType.F32);
                backend.Mul(glu, gateSilu, value);
                gateSilu.Dispose();
                value.Dispose();

                Tensor blockOut = new Tensor(new TensorShape(count, pix, nh), DType.F32);
                backend.BatchedMatMul(blockOut, glu, w3);
                glu.Dispose();
                w3.Dispose();
                outTiles[t] = blockOut;
            }
            xn.Dispose();

            Tensor blockOutFull;
            if (tileCount == 1)
            {
                blockOutFull = outTiles[0];
            }
            else
            {
                blockOutFull = new Tensor(new TensorShape(bn, pix, nh), DType.F32);
                backend.Concat(blockOutFull, outTiles, dim: 0);
                for (int t = 0; t < tileCount; t++) outTiles[t].Dispose();
            }

            Tensor xNew = new Tensor(x.Shape, DType.F32);
            backend.Add(xNew, x, blockOutFull);
            blockOutFull.Dispose();
            if (!ReferenceEquals(x, embed)) x.Dispose();
            x = xNew;
        }

        // ── Final RMSNorm, fold patch tokens back to image layout, final conv → x0 [B, 3, H, W]. ──
        Tensor xnFinal = new Tensor(x.Shape, DType.F32);
        backend.RmsNorm(xnFinal, x, _finalNormScale!, NormEps);
        if (!ReferenceEquals(x, embed)) x.Dispose();

        Tensor features = new Tensor(new TensorShape(batch, nh, height, width), DType.F32);
        if (batch == 1)
        {
            backend.UnpatchifyTokens(features, xnFinal, nh, hPacked, wPacked, p, innerChannelFastest: true);
        }
        else
        {
            UnpatchifyBatched(backend, features, xnFinal, batch, seqLen, nh, hPacked, wPacked, p, height, width);
        }
        xnFinal.Dispose();

        int outChannels = (int)_finalConvWeight.Shape[0];
        Tensor x0 = new Tensor(new TensorShape(batch, outChannels, height, width), DType.F32);
        backend.Conv2D(x0, features, _finalConvWeight, _finalConvBias, 1, 1, _finalConvPad, _finalConvPad);
        features.Dispose();
        return x0;
    }

    /// <summary>Slices one generated-weight chunk out of the packed params, L2-normalizes it along the input dim
    /// (transpose → RMSNorm with 1/sqrt(inDim) scale → transpose back), returning the [bn, inDim, outDim] batched
    /// weight <c>BatchedMatMul</c> consumes.</summary>
    private static Tensor NormalizedChunk(IBackend backend, Tensor genParams, int offset, int inDim, int outDim,
        long bn, Tensor l2Scale)
    {
        Tensor raw = new Tensor(new TensorShape(bn, inDim, outDim), DType.F32);
        backend.SliceLastDim(raw, genParams, offset);
        Tensor flipped = new Tensor(new TensorShape(bn, outDim, inDim), DType.F32);
        backend.Permute0213(flipped, raw, inDim, outDim, 1);
        raw.Dispose();
        Tensor normed = new Tensor(flipped.Shape, DType.F32);
        backend.RmsNorm(normed, flipped, l2Scale, L2Eps);
        flipped.Dispose();
        Tensor result = new Tensor(new TensorShape(bn, inDim, outDim), DType.F32);
        backend.Permute0213(result, normed, outDim, inDim, 1);
        normed.Dispose();
        return result;
    }

    /// <summary>Batch>1 unpatchify fallback: per-element device row slice + unpatchify, concatenated on dim 0.</summary>
    private static void UnpatchifyBatched(IBackend backend, Tensor features, Tensor tokens, int batch, int seqLen,
        int nh, int hPacked, int wPacked, int p, int height, int width)
    {
        int pix = p * p;
        Tensor[] parts = new Tensor[batch];
        for (int b = 0; b < batch; b++)
        {
            Tensor slice = new Tensor(new TensorShape((long)seqLen * pix, nh), DType.F32);
            backend.SliceRows(slice, tokens, b * seqLen * pix);
            Tensor part = new Tensor(new TensorShape(1, nh, height, width), DType.F32);
            backend.UnpatchifyTokens(part, slice, nh, hPacked, wPacked, p, innerChannelFastest: true);
            slice.Dispose();
            parts[b] = part;
        }
        backend.Concat(features, parts, dim: 0);
        for (int b = 0; b < batch; b++) parts[b].Dispose();
    }

    /// <summary>Folds the per-pixel embed Linear into a stride-P Conv2d: weight [pix·nh, 3, P, P] places the
    /// Linear's RGB columns at each pixel's kernel position; the positional-feature term (constant per pixel
    /// position) collapses into the bias: <c>bias[pixIdx·nh + j] = b[j] + Σ_f W[j, 3+f]·posFeat[pixIdx, f]</c>.
    /// The conv output channel order (pixIdx-major, nh-minor) matches the [N, P², nh] token layout after the
    /// channel/seq flip in <see cref="EmbedPixels"/>.</summary>
    private void BuildFoldedEmbedConv()
    {
        int p = _patchSize;
        int pix = p * p;
        int nh = _nerfHidden;
        int posFeatures = _maxFreqs * _maxFreqs;
        float[] posFeat = BuildPositionalFeatures(p, _maxFreqs);

        Tensor convWeight = new Tensor(new TensorShape((long)pix * nh, 3, p, p), DType.F32);
        Tensor convBias = new Tensor(new TensorShape((long)pix * nh), DType.F32);
        float* w = (float*)convWeight.DataPointer;
        float* b = (float*)convBias.DataPointer;
        float* embedW = (float*)_embedWeight!.DataPointer;
        float* embedB = (float*)_embedBias!.DataPointer;
        int embedIn = 3 + posFeatures;

        for (int pixIdx = 0; pixIdx < pix; pixIdx++)
        {
            int py = pixIdx / p;
            int px = pixIdx % p;
            for (int j = 0; j < nh; j++)
            {
                long o = (long)pixIdx * nh + j;
                for (int c = 0; c < 3; c++)
                    w[((o * 3 + c) * p + py) * p + px] = embedW[(long)j * embedIn + c];
                float acc = embedB[j];
                for (int f = 0; f < posFeatures; f++)
                    acc += embedW[(long)j * embedIn + 3 + f] * posFeat[pixIdx * posFeatures + f];
                b[o] = acc;
            }
        }
        _embedConvWeight = convWeight;
        _embedConvBias = convBias;
    }

    /// <summary>Builds the constant 1/sqrt(dim) RMSNorm scale vectors used for the exact L2 normalize.</summary>
    private void BuildL2Scales()
    {
        int inner = _nerfHidden * _mlpRatio;
        _l2ScaleNh = FilledVector(_nerfHidden, 1.0f / MathF.Sqrt(_nerfHidden));
        _l2ScaleInner = FilledVector(inner, 1.0f / MathF.Sqrt(inner));
    }

    private static Tensor FilledVector(int dim, float value)
    {
        Tensor t = new Tensor(new TensorShape(dim), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < dim; i++) p[i] = value;
        return t;
    }

    /// <summary>Damped separable cosine (DCT-like) positional basis over the P×P patch, matching ComfyUI's
    /// <c>NerfEmbedder.fetch_pos</c>: feature[u·F+v](y, x) = cos(π·u·x)·cos(π·v·y) / (1 + u·v) with
    /// y, x ∈ linspace(0, 1, P). Note the u↔x / v↔y pairing and the 1/(1+u·v) damping coefficient.</summary>
    public static float[] BuildPositionalFeatures(int patchSize, int maxFreqs)
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
                    float cx = MathF.Cos(MathF.PI * u * xPos);
                    for (int v = 0; v < maxFreqs; v++)
                        result[baseIdx + u * maxFreqs + v] = cx * MathF.Cos(MathF.PI * v * yPos) / (1.0f + u * v);
                }
            }
        }
        return result;
    }

    private static Tensor AsF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    /// <summary>Releases weight references.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _embedWeight = null;
            _embedBias = null;
            _embedConvWeight = null;
            _embedConvBias = null;
            _l2ScaleNh = null;
            _l2ScaleInner = null;
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
