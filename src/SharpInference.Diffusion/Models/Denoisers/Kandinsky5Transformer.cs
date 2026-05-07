using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Kandinsky 5 text-to-image transformer (<c>Kandinsky5Transformer3DModel</c>, image variant).
/// Mirrors <c>diffusers/models/transformers/transformer_kandinsky.py</c> for the t2i forward path
/// (no video duration, no fractal sparse attention).
///
/// Pipeline:
/// <list type="number">
/// <item>Project the Qwen2.5-VL sequence embeddings <c>[B, S_t, 3584]</c> through <c>Linear → LayerNorm</c>
/// to <c>[B, S_t, model_dim]</c>.</item>
/// <item>Project the CLIP-L pooled embeddings <c>[B, 768]</c> through <c>Linear → LayerNorm</c> to
/// <c>[B, time_dim]</c>; add to the timestep embedding.</item>
/// <item>Patch-embed the latent <c>[B, 1, h, w, in_visual_dim]</c> via reshape + <c>Linear(p_t·p_h·p_w·c → model_dim)</c>.</item>
/// <item>Run <c>num_text_blocks</c> encoder blocks on the text stream with 1D RoPE.</item>
/// <item>Run <c>num_visual_blocks</c> decoder blocks on the visual stream with 3D RoPE for self-attn
/// and cross-attention to the text stream output.</item>
/// <item>Final <c>OutLayer</c>: 2-param modulation (shift, scale) → non-affine LayerNorm → modulate →
/// <c>Linear(model_dim → p_t·p_h·p_w·out_visual_dim)</c> → reshape back to <c>[B, 1, H, W, out_visual_dim]</c>.</item>
/// </list>
///
/// Tensor layout note: diffusers uses NHWC for visual tokens (channels last). Our pipeline produces
/// BCHW latents; the patch-embed step transposes channel-last during reshape and the final
/// <see cref="ToBchw"/> step converts back so the rest of SharpInference's VAE/scheduler stack stays
/// in the same NCHW convention as Flux/SD3/Z-Image.</summary>
public sealed unsafe class Kandinsky5Transformer : IDisposable
{
    private readonly Kandinsky5Config _config;
    private readonly Kandinsky5EncoderBlock[] _textBlocks;
    private readonly Kandinsky5DecoderBlock[] _visualBlocks;
    private readonly Kandinsky5Rope _textRope;
    private readonly Kandinsky5Rope _visualRope;

    private int _disposed;

    private Tensor? _textProjWeight, _textProjBias;
    private Tensor? _textNormWeight, _textNormBias;

    private Tensor? _pooledProjWeight, _pooledProjBias;
    private Tensor? _pooledNormWeight, _pooledNormBias;

    private Tensor? _timeIn1Weight, _timeIn1Bias;
    private Tensor? _timeIn2Weight, _timeIn2Bias;

    private Tensor? _visualProjWeight, _visualProjBias;

    private Tensor? _outModWeight, _outModBias;
    private Tensor? _outProjWeight, _outProjBias;

    public Kandinsky5Transformer(Kandinsky5Config config)
    {
        _config = config;

        _textBlocks = new Kandinsky5EncoderBlock[config.NumTextBlocks];
        for (int i = 0; i < config.NumTextBlocks; i++)
            _textBlocks[i] = new Kandinsky5EncoderBlock(
                config.ModelDim, config.TimeDim, config.FfDim, config.HeadDim, config.QkNormEps);

        _visualBlocks = new Kandinsky5DecoderBlock[config.NumVisualBlocks];
        for (int i = 0; i < config.NumVisualBlocks; i++)
            _visualBlocks[i] = new Kandinsky5DecoderBlock(
                config.ModelDim, config.TimeDim, config.FfDim, config.HeadDim, config.QkNormEps);

        _textRope = new Kandinsky5Rope(config.HeadDim, config.RopeMaxPeriod);
        _visualRope = new Kandinsky5Rope(config.HeadDim, config.RopeMaxPeriod);
    }

    /// <summary>Loads weights using the diffusers state-dict naming. See
    /// <see cref="Kandinsky5CheckpointConverter"/> for the canonical key set.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _textProjWeight = weights["text_embeddings.in_layer.weight"];
        _textProjBias   = weights["text_embeddings.in_layer.bias"];
        _textNormWeight = weights["text_embeddings.norm.weight"];
        _textNormBias   = weights["text_embeddings.norm.bias"];

        _pooledProjWeight = weights["pooled_text_embeddings.in_layer.weight"];
        _pooledProjBias   = weights["pooled_text_embeddings.in_layer.bias"];
        _pooledNormWeight = weights["pooled_text_embeddings.norm.weight"];
        _pooledNormBias   = weights["pooled_text_embeddings.norm.bias"];

        _timeIn1Weight = weights["time_embeddings.in_layer.weight"];
        _timeIn1Bias   = weights["time_embeddings.in_layer.bias"];
        _timeIn2Weight = weights["time_embeddings.out_layer.weight"];
        _timeIn2Bias   = weights["time_embeddings.out_layer.bias"];

        _visualProjWeight = weights["visual_embeddings.in_layer.weight"];
        _visualProjBias   = weights["visual_embeddings.in_layer.bias"];

        for (int i = 0; i < _config.NumTextBlocks; i++)
            _textBlocks[i].LoadWeights(weights, $"text_transformer_blocks.{i}");

        for (int i = 0; i < _config.NumVisualBlocks; i++)
            _visualBlocks[i].LoadWeights(weights, $"visual_transformer_blocks.{i}");

        _outModWeight = weights["out_layer.modulation.out_layer.weight"];
        _outModBias   = weights["out_layer.modulation.out_layer.bias"];
        _outProjWeight = weights["out_layer.out_layer.weight"];
        _outProjBias   = weights["out_layer.out_layer.bias"];
    }

    /// <summary>Yields every weight tensor for GPU preloading via <c>backend.PreloadWeights</c>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head =
        [
            _textProjWeight, _textProjBias, _textNormWeight, _textNormBias,
            _pooledProjWeight, _pooledProjBias, _pooledNormWeight, _pooledNormBias,
            _timeIn1Weight, _timeIn1Bias, _timeIn2Weight, _timeIn2Bias,
            _visualProjWeight, _visualProjBias,
        ];
        for (int i = 0; i < head.Length; i++)
            if (head[i] is Tensor t) yield return t;

        for (int i = 0; i < _textBlocks.Length; i++)
            foreach (Tensor w in _textBlocks[i].EnumerateWeights()) yield return w;

        for (int i = 0; i < _visualBlocks.Length; i++)
            foreach (Tensor w in _visualBlocks[i].EnumerateWeights()) yield return w;

        Tensor?[] tail = [_outModWeight, _outModBias, _outProjWeight, _outProjBias];
        for (int i = 0; i < tail.Length; i++)
            if (tail[i] is Tensor t) yield return t;
    }

    /// <summary>Forward pass: predicts velocity for one denoising step. Inputs in BCHW layout (matches
    /// the rest of the SharpInference diffusion stack); the channel-last conversion happens internally.</summary>
    /// <param name="latent">Noisy latent <c>[B, in_visual_dim, H_lat, W_lat]</c>.</param>
    /// <param name="timestep">Scaled timestep value (sigma * 1000).</param>
    /// <param name="textEmbeds">Qwen2.5-VL sequence embeddings <c>[B, S_t, in_text_dim]</c>.</param>
    /// <param name="pooledEmbeds">CLIP-L pooled embeddings <c>[B, in_text_dim2]</c>.</param>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeds, Tensor pooledEmbeds)
    {
        ThrowIfDisposed();

        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int latH = (int)latent.Shape[2];
        int latW = (int)latent.Shape[3];

        (int pT, int pH, int pW) = _config.PatchSize;
        if (pT != 1)
            throw new InvalidOperationException(
                $"Kandinsky5Transformer (image variant) requires patch_size[0]=1; got {pT}");
        if (latH % pH != 0 || latW % pW != 0)
            throw new InvalidOperationException(
                $"Latent size {latH}x{latW} must be divisible by patch size {pH}x{pW}");

        int gridT = 1, gridH = latH / pH, gridW = latW / pW;
        int numPatches = gridT * gridH * gridW;
        int dim = _config.ModelDim;

        // ── 1. Project text embeddings ──
        Tensor textHidden = ProjectText(backend, textEmbeds);
        Kandinsky5DebugDump.Dump("text_proj", textHidden);

        // ── 2. Project pooled embeddings + timestep MLP, sum ──
        Tensor pooledHidden = ProjectPooled(backend, pooledEmbeds);
        Kandinsky5DebugDump.Dump("pooled_proj", pooledHidden);

        Tensor temb = ComputeTimestepEmbedding(backend, timestep, batch);
        AddInPlace(temb, pooledHidden);
        pooledHidden.Dispose();
        Kandinsky5DebugDump.Dump("time_embed", temb);

        // ── 3. Patch embed ──
        Tensor visualTokens = PatchEmbed(backend, latent, batch, channels, latH, latW, gridH, gridW);
        Kandinsky5DebugDump.Dump("patch_embed", visualTokens);

        // ── 4. Build RoPEs ──
        int sT = (int)textHidden.Shape[1];
        Span<int> textPositions = stackalloc int[0];
        int[] textPosArr = new int[sT];
        for (int i = 0; i < sT; i++) textPosArr[i] = i;
        _textRope.Precompute1D(textPosArr);
        _visualRope.Precompute3D(_config.AxesDims, gridT, gridH, gridW);

        // ── 5. Text encoder blocks ──
        Tensor curText = textHidden;
        for (int i = 0; i < _textBlocks.Length; i++)
        {
            Tensor next = _textBlocks[i].Forward(backend, curText, temb, _textRope);
            curText.Dispose();
            curText = next;
            Kandinsky5DebugDump.Dump($"text_block_{i}", curText);
        }

        // ── 6. Visual decoder blocks ──
        Tensor curVisual = visualTokens;
        for (int i = 0; i < _visualBlocks.Length; i++)
        {
            Tensor next = _visualBlocks[i].Forward(backend, curVisual, curText, temb, _visualRope);
            curVisual.Dispose();
            curVisual = next;
            if (Kandinsky5DebugDump.Enabled)
                Kandinsky5DebugDump.Dump($"visual_block_{i}", curVisual);
        }
        curText.Dispose();

        // ── 7. Final out layer: 2-param modulation + LayerNorm + Linear ──
        Tensor finalProj = ApplyOutLayer(backend, curVisual, temb, batch, numPatches, gridH, gridW);
        curVisual.Dispose();
        temb.Dispose();
        Kandinsky5DebugDump.Dump("out_layer", finalProj);

        // ── 8. Unpatchify back to BCHW ──
        Tensor result = Unpatchify(finalProj, batch, gridH, gridW, latH, latW);
        finalProj.Dispose();
        Kandinsky5DebugDump.DumpOutput(result);

        return result;
    }

    /// <summary>Pipeline-level debug hook: dumps the post-denoise, pre-VAE latent.</summary>
    public static void DumpFinalLatent(Tensor latent) => Kandinsky5DebugDump.Dump("final_latent", latent);

    private Tensor ProjectText(IBackend backend, Tensor textEmbeds)
    {
        int batch = (int)textEmbeds.Shape[0];
        int seqLen = (int)textEmbeds.Shape[1];
        int dim = _config.ModelDim;

        Tensor projected = new Tensor(new TensorShape(batch, seqLen, dim), DType.F32);
        backend.Linear(projected, textEmbeds, _textProjWeight!, _textProjBias);

        Tensor normed = new Tensor(new TensorShape(batch, seqLen, dim), DType.F32);
        backend.LayerNorm(normed, projected, _textNormWeight!, _textNormBias!, 1e-5f);
        projected.Dispose();
        return normed;
    }

    private Tensor ProjectPooled(IBackend backend, Tensor pooled)
    {
        int batch = (int)pooled.Shape[0];
        int outDim = _config.TimeDim;

        Tensor projected = new Tensor(new TensorShape(batch, outDim), DType.F32);
        backend.Linear(projected, pooled, _pooledProjWeight!, _pooledProjBias);

        Tensor normed = new Tensor(new TensorShape(batch, outDim), DType.F32);
        backend.LayerNorm(normed, projected, _pooledNormWeight!, _pooledNormBias!, 1e-5f);
        projected.Dispose();
        return normed;
    }

    /// <summary>Computes the time embedding per <c>Kandinsky5TimeEmbeddings</c>:
    /// <c>args = t ⊗ freqs(model_dim/2)</c> → <c>cat(cos, sin)</c> → <c>Linear(model_dim → time_dim) → SiLU →
    /// Linear(time_dim → time_dim)</c>. Note this differs from the AuraFlow timestep MLP: the sinusoidal
    /// dim is <c>model_dim</c> (not 256) and the activation is between the two linears, not after.</summary>
    private Tensor ComputeTimestepEmbedding(IBackend backend, float timestep, int batch)
    {
        int modelDim = _config.ModelDim;
        int timeDim = _config.TimeDim;
        int half = modelDim / 2;

        Tensor sin = new Tensor(new TensorShape(batch, modelDim), DType.F32);
        float* sinPtr = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int baseOff = b * modelDim;
            for (int k = 0; k < half; k++)
            {
                double freq = 1.0 / Math.Pow(_config.RopeMaxPeriod, (double)(2 * k) / modelDim);
                double angle = timestep * freq;
                sinPtr[baseOff + k] = (float)Math.Cos(angle);
                sinPtr[baseOff + half + k] = (float)Math.Sin(angle);
            }
        }

        Tensor mid = new Tensor(new TensorShape(batch, timeDim), DType.F32);
        backend.Linear(mid, sin, _timeIn1Weight!, _timeIn1Bias);
        sin.Dispose();

        Tensor activated = new Tensor(new TensorShape(batch, timeDim), DType.F32);
        backend.Silu(activated, mid);
        mid.Dispose();

        Tensor temb = new Tensor(new TensorShape(batch, timeDim), DType.F32);
        backend.Linear(temb, activated, _timeIn2Weight!, _timeIn2Bias);
        activated.Dispose();
        return temb;
    }

    /// <summary>Adds <paramref name="add"/> into <paramref name="dst"/> element-wise (both
    /// <c>[B, time_dim]</c>). Equivalent to <c>dst += add</c>.</summary>
    private static void AddInPlace(Tensor dst, Tensor add)
    {
        long n = dst.ElementCount;
        if (add.ElementCount != n)
            throw new InvalidOperationException("AddInPlace: tensor sizes differ");
        float* d = (float*)dst.DataPointer;
        float* a = (float*)add.DataPointer;
        for (long i = 0; i < n; i++)
            d[i] += a[i];
    }

    /// <summary>Patch embed for the image variant. Latent <c>[B, C, H, W]</c> in BCHW reshapes to
    /// <c>[B, gridH, gridW, p_h * p_w * C]</c> with C as the innermost axis (matching diffusers'
    /// <c>permute(0, 1, 3, 5, 2, 4, 6, 7).flatten(4, 7)</c> with duration=1), then projects via
    /// <c>visual_embeddings.in_layer</c> to <c>[B, gridH * gridW, model_dim]</c>.</summary>
    private Tensor PatchEmbed(IBackend backend, Tensor latent, int batch, int channels, int latH, int latW,
        int gridH, int gridW)
    {
        (int _, int pH, int pW) = _config.PatchSize;
        int patchInDim = pH * pW * channels;
        int numPatches = gridH * gridW;

        Tensor flat = new Tensor(new TensorShape(batch, numPatches, patchInDim), DType.F32);
        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)flat.DataPointer;

        // Reference reshape with duration=1 and channel last:
        //   [B, 1, H, W, C] → [B, 1, H/pH, pH, W/pW, pW, C]
        //   → permute (0, 1, 2, 4, 3, 5, 6) on the (h_outer, h_inner, w_outer, w_inner, c) tail
        //   → flatten(3, 6) giving [B, gridH, gridW, pH*pW*C].
        // Since our latent is BCHW (not BHWC) we need an extra "channel last" twist: at flatten time
        // diffusers' flatten(4, 7) yields innermost order (pH, pW, C). With BCHW input we walk
        // (c, py, px) in nested loops and write to (py, px, c) in the patch.
        for (int b = 0; b < batch; b++)
        {
            for (int gy = 0; gy < gridH; gy++)
            {
                for (int gx = 0; gx < gridW; gx++)
                {
                    int patchIdx = gy * gridW + gx;
                    int dstBase = (b * numPatches + patchIdx) * patchInDim;
                    for (int py = 0; py < pH; py++)
                    {
                        for (int px = 0; px < pW; px++)
                        {
                            int srcY = gy * pH + py;
                            int srcX = gx * pW + px;
                            for (int c = 0; c < channels; c++)
                            {
                                int srcIdx = ((b * channels + c) * latH + srcY) * latW + srcX;
                                int dstIdx = dstBase + (py * pW + px) * channels + c;
                                outPtr[dstIdx] = inPtr[srcIdx];
                            }
                        }
                    }
                }
            }
        }

        Tensor projected = new Tensor(new TensorShape(batch, numPatches, _config.ModelDim), DType.F32);
        backend.Linear(projected, flat, _visualProjWeight!, _visualProjBias);
        flat.Dispose();
        return projected;
    }

    /// <summary>Final output layer per <c>Kandinsky5OutLayer.forward</c>: produces 2 modulation params
    /// (shift, scale), applies non-affine LayerNorm + modulate, then projects to
    /// <c>p_h * p_w * out_visual_dim</c> per token.</summary>
    private Tensor ApplyOutLayer(IBackend backend, Tensor visual, Tensor temb, int batch, int numPatches,
        int gridH, int gridW)
    {
        int dim = _config.ModelDim;
        (int _, int pH, int pW) = _config.PatchSize;
        int patchOutDim = pH * pW * _config.OutVisualDim;

        Tensor activated = new Tensor(new TensorShape(batch, _config.TimeDim), DType.F32);
        backend.Silu(activated, temb);

        Tensor mod = new Tensor(new TensorShape(batch, 2 * dim), DType.F32);
        backend.Linear(mod, activated, _outModWeight!, _outModBias);
        activated.Dispose();

        Tensor normed = new Tensor(new TensorShape(batch, numPatches, dim), DType.F32);
        DiTUtils.LayerNormNoAffine(normed, visual, batch, numPatches, dim);

        // Apply modulation: out = norm * (1 + scale) + shift.
        // mod chunk order is (shift, scale) per `torch.chunk(self.modulation(...), 2, dim=-1)`.
        Tensor modulated = new Tensor(new TensorShape(batch, numPatches, dim), DType.F32);
        float* nPtr = (float*)normed.DataPointer;
        float* mPtr = (float*)mod.DataPointer;
        float* outPtr = (float*)modulated.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int modBase = b * 2 * dim;
            for (int s = 0; s < numPatches; s++)
            {
                int rowOff = (b * numPatches + s) * dim;
                for (int d = 0; d < dim; d++)
                {
                    float shift = mPtr[modBase + d];
                    float scale = mPtr[modBase + dim + d];
                    outPtr[rowOff + d] = nPtr[rowOff + d] * (1.0f + scale) + shift;
                }
            }
        }
        normed.Dispose();
        mod.Dispose();

        Tensor projected = new Tensor(new TensorShape(batch, numPatches, patchOutDim), DType.F32);
        backend.Linear(projected, modulated, _outProjWeight!, _outProjBias);
        modulated.Dispose();
        return projected;
    }

    /// <summary>Reverses the patch embed reshape, going from <c>[B, num_patches, p_h * p_w * C_out]</c>
    /// back to BCHW <c>[B, C_out, H, W]</c>. The innermost token order is <c>(py, px, c)</c> matching
    /// the diffusers' final permute pattern that places C last before flattening.</summary>
    private Tensor Unpatchify(Tensor patched, int batch, int gridH, int gridW, int latH, int latW)
    {
        (int _, int pH, int pW) = _config.PatchSize;
        int outChannels = _config.OutVisualDim;
        int patchOutDim = pH * pW * outChannels;
        int numPatches = gridH * gridW;

        Tensor result = new Tensor(new TensorShape(batch, outChannels, latH, latW), DType.F32);
        float* srcPtr = (float*)patched.DataPointer;
        float* dstPtr = (float*)result.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int gy = 0; gy < gridH; gy++)
            {
                for (int gx = 0; gx < gridW; gx++)
                {
                    int patchIdx = gy * gridW + gx;
                    int srcBase = (b * numPatches + patchIdx) * patchOutDim;
                    for (int py = 0; py < pH; py++)
                    {
                        for (int px = 0; px < pW; px++)
                        {
                            int dstY = gy * pH + py;
                            int dstX = gx * pW + px;
                            for (int c = 0; c < outChannels; c++)
                            {
                                int srcIdx = srcBase + (py * pW + px) * outChannels + c;
                                int dstIdx = ((b * outChannels + c) * latH + dstY) * latW + dstX;
                                dstPtr[dstIdx] = srcPtr[srcIdx];
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <summary>Releases all tensor references held by this transformer.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _textProjWeight = null; _textProjBias = null;
            _textNormWeight = null; _textNormBias = null;
            _pooledProjWeight = null; _pooledProjBias = null;
            _pooledNormWeight = null; _pooledNormBias = null;
            _timeFreqs = null;
            _timeIn1Weight = null; _timeIn1Bias = null;
            _timeIn2Weight = null; _timeIn2Bias = null;
            _visualProjWeight = null; _visualProjBias = null;
            _outModWeight = null; _outModBias = null;
            _outProjWeight = null; _outProjBias = null;
        }
        GC.SuppressFinalize(this);
    }

    /// <summary>Helper for tests: convert BCHW → channel-last (BHWC) view used internally. Unused
    /// publicly but kept for parity with diffusers' input contract for advanced callers that already
    /// have channel-last latents.</summary>
    public static Tensor ToBchw(Tensor bhwc)
    {
        int b = (int)bhwc.Shape[0];
        int h = (int)bhwc.Shape[1];
        int w = (int)bhwc.Shape[2];
        int c = (int)bhwc.Shape[3];

        Tensor result = new Tensor(new TensorShape(b, c, h, w), DType.F32);
        float* src = (float*)bhwc.DataPointer;
        float* dst = (float*)result.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int hi = 0; hi < h; hi++)
                    for (int wi = 0; wi < w; wi++)
                        dst[((bi * c + ci) * h + hi) * w + wi] = src[((bi * h + hi) * w + wi) * c + ci];
        return result;
    }
}
