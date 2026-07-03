using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>ERNIE-Image (Baidu, Apache-2.0) — single-stream DiT with **shared AdaLN modulation across all layers**. The 6-vector modulation tuple <c>(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)</c> is computed ONCE per denoising step from the timestep embedding and broadcast into every block — this saves ~300 MB of per-block AdaLN linear weights vs Flux's per-block scheme.
///
/// Architectural highlights (verified against <c>diffusers/models/transformers/transformer_ernie_image.py</c>):
/// <list type="bullet">
///   <item>Patch embed: 1×1 Conv2d-with-bias, <c>[B, in_channels=128, H, W]</c> → <c>[B, hidden, H, W]</c> → flatten spatial → <c>[B, H*W, hidden]</c>.</item>
///   <item>Text projection: <c>Linear(text_in_dim, hidden, bias=False)</c> applied to padded text embeddings <c>[B, Tmax, text_in_dim]</c>.</item>
///   <item>Timestep: <c>get_timestep_embedding(t, hidden, flip_sin_to_cos=False, downscale_freq_shift=0)</c> → <c>Linear → SiLU → Linear</c> (the diffusers <c>TimestepEmbedding</c>).</item>
///   <item>Shared modulation: <c>SiLU → Linear(hidden, 6*hidden)</c> on the timestep embedding. Output is broadcast into every block.</item>
///   <item>Sequence concat order: <c>[image_tokens; text_tokens]</c> (image first — matches <c>transformer_ernie_image.py:365</c>).</item>
///   <item>3D RoPE per <see cref="ErnieImageRope"/> with image positions offset by per-batch <c>text_lens</c>.</item>
///   <item>Padding-aware attention mask <c>[B, 1, 1, totalSeq]</c>: image tokens always valid, text valid up to <c>text_lens[b]</c>.</item>
///   <item>Final layer: <c>LayerNorm-no-affine</c> + <c>Linear(hidden, 2*hidden)</c> chunked as <c>(scale, shift)</c> (scale FIRST), then <c>x*(1+scale)+shift</c>, then <c>Linear(hidden, p² * out_channels=128)</c>.</item>
/// </list></summary>
public sealed unsafe class ErnieImageTransformer : IDisposable
{
    private readonly ErnieImageConfig _config;
    private readonly ErnieImagePatchEmbed _patchEmbed;
    private readonly ErnieImageBlock[] _blocks;
    private readonly ErnieImageRope _rope;
    private int _disposed;

    // text_proj: Linear(text_in_dim, hidden, bias=False). Optional — present only when text_in_dim != hidden.
    private Tensor? _textProjWeight;

    // time_embedding MLP: Linear(hidden, hidden) → SiLU → Linear(hidden, hidden). HuggingFace key prefix: time_embedding.{linear_1, linear_2}.
    private Tensor? _timeLinear1Weight, _timeLinear1Bias;
    private Tensor? _timeLinear2Weight, _timeLinear2Bias;

    // adaLN_modulation = Sequential(SiLU, Linear(hidden, 6*hidden)). Diffusers index 1 is the Linear (index 0 is SiLU, no params).
    private Tensor? _adaLnModulationWeight, _adaLnModulationBias;

    // final_norm.linear: Linear(hidden, 2*hidden). The norm itself has elementwise_affine=False (no params).
    private Tensor? _finalNormLinearWeight, _finalNormLinearBias;

    // final_linear: Linear(hidden, p²·out_channels=128) with bias.
    private Tensor? _finalLinearWeight, _finalLinearBias;

    public ErnieImageTransformer(ErnieImageConfig config)
    {
        _config = config;
        _patchEmbed = new ErnieImagePatchEmbed(config.InChannels, config.HiddenSize, config.PatchSize);
        _blocks = new ErnieImageBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _blocks[i] = new ErnieImageBlock(config.HiddenSize, config.NumHeads, config.FfnHiddenSize, config.RmsNormEps);
        _rope = new ErnieImageRope([config.RopeAxesDim.Text, config.RopeAxesDim.Y, config.RopeAxesDim.X], config.RopeTheta);
    }

    /// <summary>Convenience accessor for the configured architecture.</summary>
    public ErnieImageConfig Config => _config;

    /// <summary>Loads weights from a diffusers-style key dict (post-converter).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _patchEmbed.LoadWeights(weights);

        // text_proj is only present if the trained config has text_in_dim != hidden_size. Try, but don't insist.
        if (weights.TryGetValue("text_proj.weight", out Tensor? textProj))
            _textProjWeight = textProj;

        _timeLinear1Weight = weights["time_embedding.linear_1.weight"];
        weights.TryGetValue("time_embedding.linear_1.bias", out _timeLinear1Bias);
        _timeLinear2Weight = weights["time_embedding.linear_2.weight"];
        weights.TryGetValue("time_embedding.linear_2.bias", out _timeLinear2Bias);

        _adaLnModulationWeight = weights["adaLN_modulation.1.weight"];
        weights.TryGetValue("adaLN_modulation.1.bias", out _adaLnModulationBias);

        _finalNormLinearWeight = weights["final_norm.linear.weight"];
        weights.TryGetValue("final_norm.linear.bias", out _finalNormLinearBias);

        _finalLinearWeight = weights["final_linear.weight"];
        weights.TryGetValue("final_linear.bias", out _finalLinearBias);

        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(weights, $"layers.{i}");
    }

    /// <summary>Enumerates every weight tensor for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _patchEmbed.EnumerateWeights()) yield return w;
        if (_textProjWeight is not null) yield return _textProjWeight;
        if (_timeLinear1Weight is not null) yield return _timeLinear1Weight;
        if (_timeLinear1Bias is not null) yield return _timeLinear1Bias;
        if (_timeLinear2Weight is not null) yield return _timeLinear2Weight;
        if (_timeLinear2Bias is not null) yield return _timeLinear2Bias;
        if (_adaLnModulationWeight is not null) yield return _adaLnModulationWeight;
        if (_adaLnModulationBias is not null) yield return _adaLnModulationBias;
        if (_finalNormLinearWeight is not null) yield return _finalNormLinearWeight;
        if (_finalNormLinearBias is not null) yield return _finalNormLinearBias;
        if (_finalLinearWeight is not null) yield return _finalLinearWeight;
        if (_finalLinearBias is not null) yield return _finalLinearBias;
        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;
    }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="latent">VAE latent <c>[B, in_channels=128, H, W]</c>.</param>
    /// <param name="timestep">Scalar timestep (the same value for every batch row).</param>
    /// <param name="textEmbeds">Padded text encoder hiddens <c>[B, Tmax, text_in_dim]</c>.</param>
    /// <param name="textLens">Per-batch real (non-padded) text token counts. Length must equal <c>B</c>.</param>
    /// <returns>Predicted velocity <c>[B, out_channels=128, H, W]</c>.</returns>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeds, int[] textLens)
    {
        if (latent.Shape.Rank != 4)
            throw new ArgumentException($"Latent must be 4D, got {latent.Shape}.", nameof(latent));
        if (textEmbeds.Shape.Rank != 3)
            throw new ArgumentException($"textEmbeds must be 3D [B, Tmax, dim], got {textEmbeds.Shape}.", nameof(textEmbeds));

        int batch = (int)latent.Shape[0];
        int latentH = (int)latent.Shape[2];
        int latentW = (int)latent.Shape[3];
        int hidden = _config.HiddenSize;
        int gridH = latentH / _config.PatchSize;
        int gridW = latentW / _config.PatchSize;
        int nImg = gridH * gridW;
        int textMax = (int)textEmbeds.Shape[1];
        int totalSeq = nImg + textMax;

        if (textLens.Length != batch)
            throw new ArgumentException($"textLens length {textLens.Length} != batch {batch}.", nameof(textLens));

        // ── 1. Patch-embed image latent → [B, nImg, hidden] ──
        Tensor imgTokens = _patchEmbed.Forward(backend, latent);
        ErnieImageDebugDump.Dump("patch_embed", imgTokens);

        // ── 2. Project text embeds to hidden dim → [B, Tmax, hidden] ──
        Tensor textProjected = _textProjWeight is not null
            ? ProjectText(backend, textEmbeds, batch, textMax, hidden)
            : EnsureF32(textEmbeds);
        ErnieImageDebugDump.Dump("text_proj", textProjected);

        // ── 3. Timestep embedding: get_timestep_embedding(t, hidden, flip_sin_to_cos=False, downscale_freq_shift=0) ──
        // Then Linear → SiLU → Linear to get c [B, hidden].
        Tensor c = ComputeTimestepEmbedding(backend, timestep, batch);
        ErnieImageDebugDump.Dump("time_embedding", c);

        // ── 4. Shared modulation: SiLU → Linear(hidden, 6*hidden), chunk into 6 along last dim ──
        Tensor[] mod = ComputeSharedModulation(backend, c, batch);
        // mod = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        ErnieImageDebugDump.Dump("modulation_shift_msa", mod[0]);
        ErnieImageDebugDump.Dump("modulation_scale_msa", mod[1]);
        ErnieImageDebugDump.Dump("modulation_gate_msa", mod[2]);
        ErnieImageDebugDump.Dump("modulation_shift_mlp", mod[3]);
        ErnieImageDebugDump.Dump("modulation_scale_mlp", mod[4]);
        ErnieImageDebugDump.Dump("modulation_gate_mlp", mod[5]);

        // ── 5. Concat [image, text] along sequence dim → [B, totalSeq, hidden] ──
        Tensor combined = ConcatAlongSeqDim(backend, imgTokens, textProjected, batch, nImg, textMax, hidden);
        imgTokens.Dispose();
        textProjected.Dispose();

        // ── 6. Build attention mask [B, 1, 1, totalSeq]: 1 for valid tokens, 0 for padded text ──
        Tensor attentionMask = BuildAttentionMask(batch, nImg, textMax, textLens);

        // ── 7. Build 3D RoPE freqs ──
        Tensor posIds = ErnieImageRope.BuildPositionIds(batch, textLens, gridH, gridW, textMax);
        Tensor freqs = _rope.BuildFreqs(posIds);
        posIds.Dispose();
        ErnieImageDebugDump.Dump("rope_freqs", freqs);

        // ── 8. Layer loop with broadcasted modulation ──
        Tensor x = combined;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, x,
                mod[0], mod[1], mod[2], mod[3], mod[4], mod[5],
                freqs, _rope, attentionMask);
            x.Dispose();
            x = next;
            ErnieImageDebugDump.Dump($"layers.{i}", x);
        }

        freqs.Dispose();
        attentionMask.Dispose();

        // ── 9. Slice off image tokens (front of [image, text] sequence) ──
        Tensor imageOnly = SliceImageFront(backend, x, batch, nImg, textMax, hidden);
        x.Dispose();

        // ── 10. Final layer: AdaLN-continuous(LayerNorm-no-affine + Linear → scale, shift) → modulate → Linear ──
        Tensor projected = ApplyFinalLayer(backend, imageOnly, c, batch, nImg, hidden);
        imageOnly.Dispose();
        c.Dispose();
        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();
        ErnieImageDebugDump.Dump("final_linear", projected);

        // ── 11. Reshape [B, nImg, p²·out_channels] → [B, out_channels, H, W] ──
        Tensor velocity = Unpatchify(projected, batch, gridH, gridW, _config.PatchSize, _config.OutChannels);
        projected.Dispose();
        ErnieImageDebugDump.DumpOutput(velocity);

        return velocity;
    }

    /// <summary>Projects <c>[B, Tmax, text_in_dim]</c> text embeds to <c>[B, Tmax, hidden]</c> via <c>text_proj</c> (Linear, bias=False).</summary>
    private Tensor ProjectText(IBackend backend, Tensor textEmbeds, int batch, int textMax, int hidden)
    {
        TensorShape outShape = new TensorShape(batch, textMax, hidden);
        Tensor output = new Tensor(outShape, textEmbeds.DType);
        backend.Linear(output, textEmbeds, _textProjWeight!, null);
        return output;
    }

    /// <summary>Returns input as F32 (no-op if already F32; otherwise allocates a new cast tensor). Lifetime: caller disposes.</summary>
    private static Tensor EnsureF32(Tensor t)
    {
        if (t.DType == DType.F32) return t;
        return t.CastTo(DType.F32);
    }

    /// <summary>Sinusoidal timestep embedding with diffusers' <c>flip_sin_to_cos=False, downscale_freq_shift=0</c> convention. Layout per token: <c>[sin(0..halfDim-1), cos(0..halfDim-1)]</c>.</summary>
    private static void GetTimestepEmbedding(Tensor output, float timestep, int batch, int embDim, float maxPeriod)
    {
        if (embDim % 2 != 0)
            throw new ArgumentException($"embDim {embDim} must be even.", nameof(embDim));

        int halfDim = embDim / 2;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int baseOff = b * embDim;
            for (int k = 0; k < halfDim; k++)
            {
                // exponent_k = -log(maxPeriod) * k / halfDim   (downscale_freq_shift=0 collapses to k/halfDim)
                float exponent = -MathF.Log(maxPeriod) * k / halfDim;
                float angle = timestep * MathF.Exp(exponent);
                outPtr[baseOff + k] = MathF.Sin(angle);
                outPtr[baseOff + halfDim + k] = MathF.Cos(angle);
            }
        }
    }

    /// <summary>Computes the diffusers <c>TimestepEmbedding</c> output: <c>linear_2(silu(linear_1(time_proj(t))))</c>. Result shape <c>[B, hidden]</c>.</summary>
    private Tensor ComputeTimestepEmbedding(IBackend backend, float timestep, int batch)
    {
        int hidden = _config.HiddenSize;

        TensorShape sinShape = new TensorShape(batch, hidden);
        Tensor sinEmb = new Tensor(sinShape, DType.F32);
        GetTimestepEmbedding(sinEmb, timestep, batch, hidden, maxPeriod: 10000f);

        Tensor m1 = new Tensor(sinShape, DType.F32);
        backend.Linear(m1, sinEmb, _timeLinear1Weight!, _timeLinear1Bias);
        sinEmb.Dispose();

        Tensor m1Act = new Tensor(sinShape, DType.F32);
        backend.Silu(m1Act, m1);
        m1.Dispose();

        Tensor outE = new Tensor(sinShape, DType.F32);
        backend.Linear(outE, m1Act, _timeLinear2Weight!, _timeLinear2Bias);
        m1Act.Dispose();

        return outE;
    }

    /// <summary>Shared modulation: <c>SiLU → Linear(hidden, 6*hidden)</c>, chunked into 6 along the last dim. Returned in diffusers order: <c>[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]</c> per <c>transformer_ernie_image.py:408</c>.</summary>
    private Tensor[] ComputeSharedModulation(IBackend backend, Tensor c, int batch)
    {
        int hidden = _config.HiddenSize;
        int outDim = 6 * hidden;

        Tensor act = new Tensor(c.Shape, c.DType);
        backend.Silu(act, c);

        TensorShape projShape = new TensorShape(batch, outDim);
        Tensor projected = new Tensor(projShape, DType.F32);
        backend.Linear(projected, act, _adaLnModulationWeight!, _adaLnModulationBias);
        act.Dispose();

        // Chunk [B, 6·hidden] → 6 × [B, hidden]. B=1 (the only case pipelines exercise) runs device-resident so the
        // modulation vectors stay on-device for every block's DiTUtils.Modulate; B>1 (CFG) keeps the host chunk.
        Tensor[] result = new Tensor[6];
        if (batch == 1)
        {
            for (int p = 0; p < 6; p++)
            {
                Tensor param = new Tensor(new TensorShape(batch, hidden), DType.F32);
                backend.SliceRows(param, projected, p);   // B=1: chunk p = [p·hidden, (p+1)·hidden)
                result[p] = param;
            }
            projected.Dispose();
            return result;
        }

        float* projPtr = (float*)projected.DataPointer;
        for (int p = 0; p < 6; p++)
        {
            TensorShape paramShape = new TensorShape(batch, hidden);
            Tensor param = new Tensor(paramShape, DType.F32);
            float* paramPtr = (float*)param.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                int srcOff = b * outDim + p * hidden;
                int dstOff = b * hidden;
                Buffer.MemoryCopy(projPtr + srcOff, paramPtr + dstOff, hidden * sizeof(float), hidden * sizeof(float));
            }
            result[p] = param;
        }
        projected.Dispose();
        return result;
    }

    /// <summary>Final layer (<c>ErnieImageAdaLNContinuous</c> + <c>final_linear</c>):
    /// <list type="number">
    ///   <item><c>scale, shift = chunk(linear(c), 2, dim=-1)</c> — note SCALE first, SHIFT second.</item>
    ///   <item>LayerNorm-no-affine on x.</item>
    ///   <item><c>x = x * (1 + scale) + shift</c> with conditioning broadcast over sequence.</item>
    ///   <item><c>final_linear(x)</c> projects to <c>p² * out_channels</c>.</item>
    /// </list></summary>
    private Tensor ApplyFinalLayer(IBackend backend, Tensor imageTokens, Tensor c, int batch, int seqLen, int hidden)
    {
        // 1. Linear(c) → [B, 2*hidden], chunk into (scale, shift) — scale FIRST per Python ref. B=1 (the only case
        //    pipelines exercise) slices device-resident; B>1 (CFG) keeps the host chunk (strided per batch).
        TensorShape projShape = new TensorShape(batch, 2 * hidden);
        Tensor projected = new Tensor(projShape, DType.F32);
        backend.Linear(projected, c, _finalNormLinearWeight!, _finalNormLinearBias);

        TensorShape modShape = new TensorShape(batch, hidden);
        Tensor scale = new Tensor(modShape, DType.F32);
        Tensor shift = new Tensor(modShape, DType.F32);
        if (batch == 1)
        {
            backend.SliceRows(scale, projected, 0);   // scale = chunk 0
            backend.SliceRows(shift, projected, 1);   // shift = chunk 1
        }
        else
        {
            float* projPtr = (float*)projected.DataPointer;
            float* scalePtr = (float*)scale.DataPointer;
            float* shiftPtr = (float*)shift.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                int srcBase = b * 2 * hidden;
                int dstBase = b * hidden;
                Buffer.MemoryCopy(projPtr + srcBase, scalePtr + dstBase, hidden * sizeof(float), hidden * sizeof(float));
                Buffer.MemoryCopy(projPtr + srcBase + hidden, shiftPtr + dstBase, hidden * sizeof(float), hidden * sizeof(float));
            }
        }
        projected.Dispose();

        // 2+3. LayerNorm-no-affine (eps = RmsNormEps) then AdaLN modulate x·(1+scale) + shift, broadcast over seqLen —
        //      entirely on the backend so the full image-token activation stays device-resident. Bit-identical to the
        //      old host LayerNormNoAffine + shift/scale triple-loop.
        TensorShape seqShape = new TensorShape(batch, seqLen, hidden);
        Tensor modulated = DiTUtils.NormModulate(backend, imageTokens, shift, scale, seqShape, _config.RmsNormEps);
        scale.Dispose();
        shift.Dispose();

        // 4. final_linear: [B, S, hidden] → [B, S, p² · out_channels]
        int outDim = _config.PatchSize * _config.PatchSize * _config.OutChannels;
        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor projected2 = new Tensor(outShape, DType.F32);
        backend.Linear(projected2, modulated, _finalLinearWeight!, _finalLinearBias);
        modulated.Dispose();

        return projected2;
    }

    /// <summary>Concatenates <c>[B, nImg, hidden]</c> + <c>[B, Tmax, hidden]</c> along the seq dim, image FIRST.
    /// Device-resident (<see cref="IBackend.Concat"/>, dim=1) so the joined activation never leaves the GPU between
    /// the patch-embed/text-proj GEMMs and the block loop.</summary>
    private static Tensor ConcatAlongSeqDim(IBackend backend, Tensor img, Tensor text, int batch, int nImg, int textMax, int hidden)
    {
        int totalSeq = nImg + textMax;
        TensorShape outShape = new TensorShape(batch, totalSeq, hidden);
        Tensor output = new Tensor(outShape, img.DType);

        // If text dtype != img dtype, Concat would mis-copy; ensure both are the same dtype upstream (ProjectText
        // already preserves text's dtype which may not match img's dtype — but in practice both are F32 since the
        // patch embed conv currently operates on F32 latents and ProjectText doesn't change dtype).
        if (text.DType != img.DType)
            throw new InvalidOperationException($"Concat dtype mismatch: img={img.DType} text={text.DType}.");

        backend.Concat(output, new Tensor[] { img, text }, 1);
        return output;
    }

    /// <summary>Slices the image-front section <c>[B, totalSeq, hidden]</c> → <c>[B, nImg, hidden]</c>. B=1 (the only
    /// case pipelines exercise) runs device-resident via <see cref="IBackend.SliceRows"/> — the image tokens are the
    /// contiguous <c>nImg·hidden</c> prefix — so the final-layer input never leaves the GPU. B>1 (CFG) keeps the host
    /// copy because each batch's image prefix is strided by <c>totalSeq</c>.</summary>
    private static Tensor SliceImageFront(IBackend backend, Tensor combined, int batch, int nImg, int textMax, int hidden)
    {
        int totalSeq = nImg + textMax;
        TensorShape outShape = new TensorShape(batch, nImg, hidden);
        Tensor output = new Tensor(outShape, combined.DType);

        if (batch == 1)
        {
            backend.SliceRows(output, combined, 0);   // first nImg·hidden elements = image prefix
            return output;
        }

        float* srcPtr = (float*)combined.DataPointer;
        float* dstPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long bytes = (long)nImg * hidden * sizeof(float);
            Buffer.MemoryCopy(srcPtr + (long)b * totalSeq * hidden, dstPtr + (long)b * nImg * hidden, bytes, bytes);
        }
        return output;
    }

    /// <summary>Builds the additive key-padding attention mask <c>[B, 1, totalSeq, totalSeq]</c> (broadcasts over heads
    /// in SDPA): 0 for valid keys, large-negative for padded text keys; same for every query row. Image keys are always
    /// valid. NOTE: this is tiled across the query (Sq) dimension on purpose — a <c>[B,1,1,Skv]</c> mask's element count
    /// (<c>B·Skv</c>) matches NEITHER SDPA mask branch (<c>Sq·Skv</c> / <c>B·H·Sq·Skv</c>) and was silently dropped, so
    /// every image token attended to the padded text keys → washed-out/hazy output. With Sq tiled, B=1 hits the existing
    /// <c>Sq·Skv</c> branch and B>1 (CFG) hits the new <c>B·Sq·Skv</c> branch (both broadcast over heads).</summary>
    private static Tensor BuildAttentionMask(int batch, int nImg, int textMax, int[] textLens)
    {
        int totalSeq = nImg + textMax;
        TensorShape shape = new TensorShape(batch, 1, totalSeq, totalSeq);
        Tensor mask = new Tensor(shape, DType.F32);
        mask.AsSpan<float>().Clear();   // valid positions = 0 (additive convention)
        const float negInf = -1e30f;
        float* p = (float*)mask.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int validText = Math.Clamp(textLens[b], 0, textMax);
            if (validText >= textMax) continue;   // no padding → all-zero mask for this batch
            long bBase = (long)b * totalSeq * totalSeq;
            for (int qi = 0; qi < totalSeq; qi++)
            {
                long rowOff = bBase + (long)qi * totalSeq + nImg;   // text keys start at column nImg
                for (int t = validText; t < textMax; t++) p[rowOff + t] = negInf;
            }
        }
        return mask;
    }

    /// <summary>Reshapes <c>[B, nImg, p²·out_channels]</c> back to <c>[B, out_channels, H, W]</c>. For p=1 this is just a logical permute from token-major to channel-major.</summary>
    private static Tensor Unpatchify(Tensor packed, int batch, int gridH, int gridW, int patch, int outChannels)
    {
        int height = gridH * patch;
        int width = gridW * patch;
        int seqLen = gridH * gridW;
        int patchDim = patch * patch * outChannels;

        TensorShape outShape = new TensorShape(batch, outChannels, height, width);
        Tensor output = new Tensor(outShape, packed.DType);

        float* inPtr = (float*)packed.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        // Python reference: patches.view(B, Hp, Wp, p, p, C).permute(0, 5, 1, 3, 2, 4).view(B, C, H, W).
        // Per-element index translation: for output[b, c, h, w], compute source patch (hp=h/p, wp=w/p),
        // source intra-patch (ph=h%p, pw=w%p), then source seq=hp*Wp+wp, source feature=ph*p*C+pw*C+c.
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < outChannels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int hp = h / patch;
                        int wp = w / patch;
                        int ph = h % patch;
                        int pw = w % patch;
                        long seqIdx = (long)hp * gridW + wp;
                        long featOff = ((long)ph * patch + pw) * outChannels + c;
                        long srcOff = ((long)b * seqLen + seqIdx) * patchDim + featOff;
                        long dstOff = (((long)b * outChannels + c) * height + h) * width + w;
                        outPtr[dstOff] = inPtr[srcOff];
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Disposes the transformer state. Tensors are dropped — the actual unmanaged storage is owned by the underlying mmap loader.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _textProjWeight = null;
            _timeLinear1Weight = null; _timeLinear1Bias = null;
            _timeLinear2Weight = null; _timeLinear2Bias = null;
            _adaLnModulationWeight = null; _adaLnModulationBias = null;
            _finalNormLinearWeight = null; _finalNormLinearBias = null;
            _finalLinearWeight = null; _finalLinearBias = null;
        }
        GC.SuppressFinalize(this);
    }
}
