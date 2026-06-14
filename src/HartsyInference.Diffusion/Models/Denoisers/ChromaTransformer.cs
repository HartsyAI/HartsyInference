using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Chroma diffusion transformer (<c>lodestones/Chroma</c>, ~8.9B params, 19 double + 38 single blocks).
///
/// Key architectural differences from <see cref="FluxTransformer"/>:
/// <list type="bullet">
///   <item><b>Distilled-guidance approximator</b> replaces every per-block <c>norm.linear</c>: a single shared
///         5-layer SiLU+RMSNorm-residual MLP takes <c>concat([Timesteps(t,16), Timesteps(0,16), mod_proj(arange,32)])</c>
///         and produces a <c>[B, mod_index_length, hidden]</c> table that is sliced per block.</item>
///   <item><b>T5-only conditioning</b> (no CLIP pooled, no guidance scalar). T5-XXL hidden dim 4096.</item>
///   <item>Pruned <c>ChromaAdaLayerNormZero{Pruned,SinglePruned}</c> blocks consume modulation rows directly.</item>
///   <item>Optional <b>per-token attention mask</b> propagated through every block (Flux ignores masks).</item>
///   <item>Final norm <c>ChromaAdaLayerNormContinuousPruned</c> takes <c>[B, 2, hidden]</c> from the last two rows of
///         the modulation table — chunk order is <c>[scale, shift]</c> (row 0 scale, row 1 shift). No checkpoint linear.</item>
/// </list>
///
/// Reference: <c>diffusers/models/transformers/transformer_chroma.py:423-624</c>.</summary>
public sealed unsafe class ChromaTransformer : IDisposable
{
    private readonly ChromaConfig _config;
    private readonly ChromaCombinedTimestepEmbeddings _timestepEmbed;
    private readonly ChromaApproximator _approximator;
    private readonly ChromaDoubleStreamBlock[] _doubleBlocks;
    private readonly ChromaSingleStreamBlock[] _singleBlocks;
    private readonly FluxRope _rope;

    // x_embedder: Linear(in_channels=64, hidden_size=3072, bias=True)
    private Tensor? _xEmbedWeight, _xEmbedBias;

    // context_embedder: Linear(joint_attention_dim=4096, hidden_size=3072, bias=True)
    private Tensor? _contextEmbedWeight, _contextEmbedBias;

    // proj_out: Linear(hidden_size, patch_size² * out_channels=64, bias=True)
    private Tensor? _projOutWeight, _projOutBias;

    private int _disposed;

    /// <summary>Creates a Chroma transformer from configuration.</summary>
    public ChromaTransformer(ChromaConfig config)
    {
        _config = config;

        _timestepEmbed = new ChromaCombinedTimestepEmbeddings(
            numChannels: config.ApproximatorNumChannels / 4,
            modIndexLength: config.ModIndexLength);

        _approximator = new ChromaApproximator(
            inDim: config.ApproximatorNumChannels,
            outDim: config.HiddenSize,
            hiddenDim: config.ApproximatorHiddenDim,
            numLayers: config.ApproximatorLayers);

        _doubleBlocks = new ChromaDoubleStreamBlock[config.Depth];
        for (int i = 0; i < config.Depth; i++)
        {
            _doubleBlocks[i] = new ChromaDoubleStreamBlock(
                config.HiddenSize, config.NumHeads, config.HeadDim);
        }

        _singleBlocks = new ChromaSingleStreamBlock[config.DepthSingleBlocks];
        for (int i = 0; i < config.DepthSingleBlocks; i++)
        {
            _singleBlocks[i] = new ChromaSingleStreamBlock(
                config.HiddenSize, config.NumHeads, config.HeadDim);
        }

        // Same axes/theta as Flux. Chroma reuses FluxPosEmbed verbatim.
        _rope = new FluxRope([16, 56, 56], 10000);
    }

    /// <summary>Loads all transformer weights from named tensors using diffusers naming (post-conversion).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
        => LoadWeightsInternal(weights, requireImageProjections: true);

    /// <summary>Weight-loading core shared with <see cref="ChromaRadianceTransformer"/>. Radiance checkpoints have
    /// no <c>x_embedder.*</c> / <c>proj_out.*</c> (replaced by conv patchify + NeRF head), so those lookups become
    /// optional when <paramref name="requireImageProjections"/> is false.</summary>
    internal void LoadWeightsInternal(IReadOnlyDictionary<string, Tensor> weights, bool requireImageProjections)
    {
        if (requireImageProjections)
        {
            _xEmbedWeight = weights["x_embedder.weight"];
            _xEmbedBias = weights["x_embedder.bias"];
            _projOutWeight = weights["proj_out.weight"];
            _projOutBias = weights["proj_out.bias"];
        }
        else
        {
            weights.TryGetValue("x_embedder.weight", out _xEmbedWeight);
            weights.TryGetValue("x_embedder.bias", out _xEmbedBias);
            weights.TryGetValue("proj_out.weight", out _projOutWeight);
            weights.TryGetValue("proj_out.bias", out _projOutBias);
        }

        _contextEmbedWeight = weights["context_embedder.weight"];
        _contextEmbedBias = weights["context_embedder.bias"];

        _approximator.LoadWeights(weights);

        for (int i = 0; i < _config.Depth; i++)
            _doubleBlocks[i].LoadWeights(weights, $"transformer_blocks.{i}");

        for (int i = 0; i < _config.DepthSingleBlocks; i++)
            _singleBlocks[i].LoadWeights(weights, $"single_transformer_blocks.{i}");
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_xEmbedWeight is not null) yield return _xEmbedWeight;
        if (_xEmbedBias is not null) yield return _xEmbedBias;
        if (_contextEmbedWeight is not null) yield return _contextEmbedWeight;
        if (_contextEmbedBias is not null) yield return _contextEmbedBias;
        foreach (Tensor w in _approximator.EnumerateWeights()) yield return w;
        for (int i = 0; i < _doubleBlocks.Length; i++)
            foreach (Tensor w in _doubleBlocks[i].EnumerateWeights()) yield return w;
        for (int i = 0; i < _singleBlocks.Length; i++)
            foreach (Tensor w in _singleBlocks[i].EnumerateWeights()) yield return w;
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
    }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="packedLatent">Packed latent tokens [B, imgSeqLen, 64].</param>
    /// <param name="encoderHidden">T5 text embeddings [B, txtSeqLen, 4096].</param>
    /// <param name="timestep">Timestep in [0, 1] (caller should pass <c>t/1000</c> to match diffusers).</param>
    /// <param name="txtSeqLen">Number of text tokens (used to build position ids and to split the joint sequence).</param>
    /// <param name="hPacked">Packed image height (latent_h / 2).</param>
    /// <param name="wPacked">Packed image width (latent_w / 2).</param>
    /// <param name="attentionMask">Optional [B, txtSeqLen] T5 attention mask. The transformer extends it with all-ones for image tokens to form the [B, txtSeqLen+imgSeqLen] mask passed to every block.</param>
    public Tensor Forward(
        IBackend backend,
        Tensor packedLatent,
        Tensor encoderHidden,
        float timestep,
        int txtSeqLen,
        int hPacked,
        int wPacked,
        Tensor? attentionMask)
    {
        int batch = (int)packedLatent.Shape[0];
        int imgSeqLen = (int)packedLatent.Shape[1];
        int hidden = _config.HiddenSize;

        // ── 1. img_in: [B, imgSeqLen, 64] → [B, imgSeqLen, hidden] ──
        TensorShape imgTokShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor img = new Tensor(imgTokShape, DType.F32);
        backend.Linear(img, packedLatent, _xEmbedWeight!, _xEmbedBias);
        ChromaDebugDump.Dump("img_in", img);

        (Tensor imgOut, Tensor modTable) = ForwardCore(backend, img, encoderHidden, timestep,
            txtSeqLen, hPacked, wPacked, attentionMask);

        // ── 11. Final norm (ChromaAdaLayerNormContinuousPruned) ──
        // temb_final = modTable[:, -2:, :]  → row 0 = scale, row 1 = shift
        Tensor finalScaleShift = SliceModSlab(modTable, batch, _config.ModIndexLength - 2, rowCount: 2, hidden);
        modTable.Dispose();

        Tensor normedOut = ApplyContinuousNorm(imgOut, finalScaleShift, batch, imgSeqLen, hidden);
        finalScaleShift.Dispose();
        imgOut.Dispose();
        ChromaDebugDump.Dump("img_post_norm_out", normedOut);

        // ── 12. proj_out: [B, imgSeqLen, hidden] → [B, imgSeqLen, patch_size² * out_channels=64] ──
        int outDim = (int)_projOutWeight!.Shape[0];
        TensorShape projShape = new TensorShape(batch, imgSeqLen, outDim);
        Tensor velocity = new Tensor(projShape, DType.F32);
        backend.Linear(velocity, normedOut, _projOutWeight!, _projOutBias);
        normedOut.Dispose();

        ChromaDebugDump.DumpOutput(velocity);
        return velocity;
    }

    /// <summary>Backbone forward shared with <see cref="ChromaRadianceTransformer"/>: approximator + double/single
    /// blocks, from already-embedded image tokens to pre-final-norm image tokens. Consumes (disposes)
    /// <paramref name="img"/>; the caller owns both returned tensors. <c>modTable</c> is returned so classic Chroma
    /// can slice the final norm rows — Radiance disposes it unused (its NeRF head replaces <c>final_layer</c>).</summary>
    internal (Tensor imgTokens, Tensor modTable) ForwardCore(
        IBackend backend,
        Tensor img,
        Tensor encoderHidden,
        float timestep,
        int txtSeqLen,
        int hPacked,
        int wPacked,
        Tensor? attentionMask)
    {
        int batch = (int)img.Shape[0];
        int imgSeqLen = (int)img.Shape[1];
        int hidden = _config.HiddenSize;
        int totalSeqLen = txtSeqLen + imgSeqLen;
        int numDoubles = _config.Depth;
        int numSingles = _config.DepthSingleBlocks;

        // ── 2. Timestep × 1000 (diffusers does this on the input side; the embedding sees [0, 1000]) ──
        float scaledTimestep = timestep * 1000.0f;

        // ── 3. Build approximator input then run the MLP ──
        Tensor modInput = _timestepEmbed.Forward(scaledTimestep, batch);
        ChromaDebugDump.Dump("mod_input", modInput);
        Tensor modTable = _approximator.Forward(backend, modInput);
        modInput.Dispose();
        ChromaDebugDump.Dump("mod_table", modTable);

        // ── 4. context_embedder: [B, txtSeqLen, 4096] → [B, txtSeqLen, hidden] ──
        TensorShape txtTokShape = new TensorShape(batch, txtSeqLen, hidden);
        Tensor txt = new Tensor(txtTokShape, DType.F32);
        backend.Linear(txt, encoderHidden, _contextEmbedWeight!, _contextEmbedBias);
        ChromaDebugDump.Dump("txt_in", txt);

        // ── 5. RoPE for joint sequence (text first, then image tokens) ──
        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        _rope.Precompute(posIds);
        posIds.Dispose();

        // ── 6. Extend attention mask to cover image tokens (all-ones) ──
        Tensor? joinedMask = attentionMask is not null
            ? ExtendMaskWithImageOnes(attentionMask, batch, txtSeqLen, imgSeqLen)
            : null;

        // ── 7. Double-stream blocks ──
        // Modulation table layout (matches transformer_chroma.py:546-557):
        //   rows [0,           3 * numSingles)            → single block mods (3 each, used later)
        //   rows [3*numSingles, 3*numSingles + 6*numDoubles) → double-block IMG mods (6 each)
        //   rows [3*numSingles + 6*numDoubles, 3*numSingles + 12*numDoubles) → double-block TXT mods
        //   rows [..., -2]                                 → final norm (scale, shift)
        int imgOffset = 3 * numSingles;
        int txtOffset = imgOffset + 6 * numDoubles;

        for (int i = 0; i < numDoubles; i++)
        {
            int imgRow = imgOffset + 6 * i;
            int txtRow = txtOffset + 6 * i;
            Tensor doubleTemb = BuildDoubleBlockTemb(modTable, batch, imgRow, txtRow, hidden);
            ChromaDebugDump.Dump($"double_{i}_temb", doubleTemb);

            (Tensor newImg, Tensor newTxt) = _doubleBlocks[i].Forward(backend, img, txt, doubleTemb, _rope, joinedMask);
            doubleTemb.Dispose();

            img.Dispose();
            txt.Dispose();
            img = newImg;
            txt = newTxt;

            ChromaDebugDump.Dump($"double_{i}_img_out", img);
            ChromaDebugDump.Dump($"double_{i}_txt_out", txt);
        }

        // ── 8. Concatenate [txt, img] for single-stream processing ──
        TensorShape concatShape = new TensorShape(batch, totalSeqLen, hidden);
        Tensor combined = new Tensor(concatShape, DType.F32);
        ConcatTextImage(combined, txt, img, batch, txtSeqLen, imgSeqLen, hidden);
        img.Dispose();
        txt.Dispose();

        // ── 9. Single-stream blocks ──
        for (int i = 0; i < numSingles; i++)
        {
            int singleRow = 3 * i;
            Tensor singleTemb = SliceModSlab(modTable, batch, singleRow, rowCount: 3, hidden);
            ChromaDebugDump.Dump($"single_{i}_temb", singleTemb);

            Tensor newCombined = _singleBlocks[i].Forward(backend, combined, singleTemb, _rope, joinedMask);
            singleTemb.Dispose();
            combined.Dispose();
            combined = newCombined;

            ChromaDebugDump.Dump($"single_{i}_out", combined);
        }

        joinedMask?.Dispose();

        // ── 10. Strip text prefix → image tail ──
        TensorShape imgOutShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor imgOut = new Tensor(imgOutShape, DType.F32);
        ExtractImageTokens(imgOut, combined, batch, txtSeqLen, imgSeqLen, hidden);
        combined.Dispose();
        ChromaDebugDump.Dump("img_pre_norm_out", imgOut);

        return (imgOut, modTable);
    }

    /// <summary>Convenience pass-through that hooks the static debug dumper for the final latent.</summary>
    public static void DumpFinalLatent(Tensor latent) => ChromaDebugDump.Dump("final_latent", latent);

    /// <summary>Extends a [B, txtSeqLen] mask with all-ones for image tokens, returning [B, txtSeqLen+imgSeqLen].</summary>
    private static Tensor ExtendMaskWithImageOnes(Tensor txtMask, int batch, int txtSeqLen, int imgSeqLen)
    {
        int total = txtSeqLen + imgSeqLen;
        TensorShape shape = new TensorShape(batch, total);
        Tensor output = new Tensor(shape, DType.F32);

        float* srcPtr = (float*)txtMask.DataPointer;
        float* dstPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < txtSeqLen; s++)
                dstPtr[b * total + s] = srcPtr[b * txtSeqLen + s];
            for (int s = 0; s < imgSeqLen; s++)
                dstPtr[b * total + txtSeqLen + s] = 1.0f;
        }
        return output;
    }

    /// <summary>Builds the [B, 12, hidden] modulation slab consumed by a double block by stacking the 6 IMG rows
    /// followed by the 6 TXT rows. Mirrors <c>torch.cat((pooled[:, img:img+6], pooled[:, txt:txt+6]), dim=1)</c>.</summary>
    private static Tensor BuildDoubleBlockTemb(Tensor modTable, int batch, int imgRow, int txtRow, int hidden)
    {
        int totalRows = (int)modTable.Shape[1];
        TensorShape shape = new TensorShape(batch, 12, hidden);
        Tensor output = new Tensor(shape, DType.F32);

        float* srcPtr = (float*)modTable.DataPointer;
        float* dstPtr = (float*)output.DataPointer;

        long rowBytes = (long)hidden * sizeof(float);
        for (int b = 0; b < batch; b++)
        {
            for (int r = 0; r < 6; r++)
            {
                long src = ((long)b * totalRows + (imgRow + r)) * hidden;
                long dst = ((long)b * 12 + r) * hidden;
                Buffer.MemoryCopy(srcPtr + src, dstPtr + dst, rowBytes, rowBytes);
            }
            for (int r = 0; r < 6; r++)
            {
                long src = ((long)b * totalRows + (txtRow + r)) * hidden;
                long dst = ((long)b * 12 + (6 + r)) * hidden;
                Buffer.MemoryCopy(srcPtr + src, dstPtr + dst, rowBytes, rowBytes);
            }
        }
        return output;
    }

    /// <summary>Slices a contiguous block of <paramref name="rowCount"/> rows out of the modulation table.</summary>
    private static Tensor SliceModSlab(Tensor modTable, int batch, int rowStart, int rowCount, int hidden)
    {
        int totalRows = (int)modTable.Shape[1];
        TensorShape shape = new TensorShape(batch, rowCount, hidden);
        Tensor output = new Tensor(shape, DType.F32);

        float* srcPtr = (float*)modTable.DataPointer;
        float* dstPtr = (float*)output.DataPointer;
        long rowBytes = (long)hidden * sizeof(float);

        for (int b = 0; b < batch; b++)
        {
            long src = ((long)b * totalRows + rowStart) * hidden;
            long dst = (long)b * rowCount * hidden;
            long copyBytes = (long)rowCount * rowBytes;
            Buffer.MemoryCopy(srcPtr + src, dstPtr + dst, copyBytes, copyBytes);
        }
        return output;
    }

    /// <summary>Applies <c>ChromaAdaLayerNormContinuousPruned</c>: <c>output = layernorm(x) * (1 + scale) + shift</c>
    /// where <c>(scale, shift)</c> come from <c>temb.flatten(1, 2).chunk(2, dim=1)</c>: row 0 of <paramref name="temb"/>
    /// is scale and row 1 is shift (each [B, hidden]). LayerNorm has no affine parameters.</summary>
    private static Tensor ApplyContinuousNorm(Tensor x, Tensor temb, int batch, int seqLen, int hidden)
    {
        TensorShape outShape = new TensorShape(batch, seqLen, hidden);
        Tensor output = new Tensor(outShape, DType.F32);

        // First normalize x in place (no affine).
        Tensor normed = new Tensor(outShape, DType.F32);
        DiTUtils.LayerNormNoAffine(normed, x, batch, seqLen, hidden);

        float* nPtr = (float*)normed.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* tembPtr = (float*)temb.DataPointer;
        // temb is [B, 2, hidden] in (SHIFT, SCALE) order. Per ComfyUI's
        // `comfy/ldm/chroma/layers.py:LastLayer.forward`:
        //   shift, scale = vec  # vec = (mod_vectors[:, -2:-1, :], mod_vectors[:, -1:, :])
        //   x = addcmul(shift, 1 + scale, norm_final(x))   == norm_final(x) * (1 + scale) + shift
        // Row -2 of the 344-row modulation table is SHIFT, row -1 is SCALE. Our previous code had this
        // swapped (treating row 0 of the 2-row slice as scale, row 1 as shift), which scrambled the
        // final-layer modulation and added a fixed-per-image noise overlay even when the rest of the
        // pipeline was correct.
        for (int b = 0; b < batch; b++)
        {
            int shiftBase = b * 2 * hidden + 0 * hidden;
            int scaleBase = b * 2 * hidden + 1 * hidden;
            for (int s = 0; s < seqLen; s++)
            {
                int seqBase = (b * seqLen + s) * hidden;
                for (int d = 0; d < hidden; d++)
                {
                    float scale = tembPtr[scaleBase + d];
                    float shift = tembPtr[shiftBase + d];
                    outPtr[seqBase + d] = nPtr[seqBase + d] * (1.0f + scale) + shift;
                }
            }
        }

        normed.Dispose();
        return output;
    }

    /// <summary>Concatenates [B, txtSeqLen, D] and [B, imgSeqLen, D] along the sequence dim → [B, txt+img, D].</summary>
    private static void ConcatTextImage(Tensor output, Tensor txt, Tensor img,
        int batch, int txtSeqLen, int imgSeqLen, int hidden)
    {
        float* txtPtr = (float*)txt.DataPointer;
        float* imgPtr = (float*)img.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int totalSeqLen = txtSeqLen + imgSeqLen;

        for (int b = 0; b < batch; b++)
        {
            long txtBytes = (long)txtSeqLen * hidden * sizeof(float);
            long imgBytes = (long)imgSeqLen * hidden * sizeof(float);

            Buffer.MemoryCopy(
                txtPtr + (long)b * txtSeqLen * hidden,
                outPtr + (long)b * totalSeqLen * hidden,
                txtBytes, txtBytes);

            Buffer.MemoryCopy(
                imgPtr + (long)b * imgSeqLen * hidden,
                outPtr + (long)b * totalSeqLen * hidden + (long)txtSeqLen * hidden,
                imgBytes, imgBytes);
        }
    }

    /// <summary>Extracts image tokens (the tail) from concatenated [B, txt+img, D] → [B, img, D].</summary>
    private static void ExtractImageTokens(Tensor output, Tensor combined, int batch, int txtSeqLen, int imgSeqLen, int hidden)
    {
        float* inPtr = (float*)combined.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int totalSeqLen = txtSeqLen + imgSeqLen;

        for (int b = 0; b < batch; b++)
        {
            long imgBytes = (long)imgSeqLen * hidden * sizeof(float);
            Buffer.MemoryCopy(
                inPtr + (long)b * totalSeqLen * hidden + (long)txtSeqLen * hidden,
                outPtr + (long)b * imgSeqLen * hidden,
                imgBytes, imgBytes);
        }
    }

    /// <summary>Releases all tensor references.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _xEmbedWeight = null;
            _xEmbedBias = null;
            _contextEmbedWeight = null;
            _contextEmbedBias = null;
            _projOutWeight = null;
            _projOutBias = null;
            _approximator.Dispose();
        }
        GC.SuppressFinalize(this);
    }
}
