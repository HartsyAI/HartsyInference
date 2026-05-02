using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>AuraFlow MMDiT transformer (<c>AuraFlowTransformer2DModel</c>). Patch-embeds a SDXL-VAE latent, projects Pile-T5-XL context, prepends 8 register tokens, runs <c>NumDoubleBlocks</c> dual-stream joint blocks, then concatenates <c>[txt, img]</c> and runs <c>NumSingleBlocks</c> single-stream blocks. Final layer is the AuraFlow PreFinalBlock (<c>silu(temb) → Linear → [scale, shift]</c>; note the order is opposite of SD3's <c>[shift, scale]</c>) followed by a bias-free <c>proj_out</c>.</summary>
public sealed unsafe class AuraFlowTransformer : IDisposable
{
    private readonly AuraFlowConfig _config;
    private readonly AuraFlowJointBlock[] _doubleBlocks;
    private readonly AuraFlowSingleBlock[] _singleBlocks;
    private readonly Unpatchify _unpatchify;
    private int _disposed;

    // Patch embed: a Linear (NOT Conv2d) over flattened patches, plus learned 2D pos_embed.
    private Tensor? _patchProjWeight;
    private Tensor? _patchProjBias;
    private Tensor? _posEmbed;
    private int _posEmbedMaxSize;
    private int _posEmbedAxis;

    // Timestep MLP (TimestepEmbedding): Linear → SiLU → Linear, with biases.
    private Tensor? _timestepLinear1Weight, _timestepLinear1Bias;
    private Tensor? _timestepLinear2Weight, _timestepLinear2Bias;

    // Context embedder: Linear(joint_attention_dim → caption_projection_dim, bias=False).
    private Tensor? _contextEmbedWeight;

    // 8 register tokens prepended to text. Shape [1, 8, inner_dim]. Loaded as F32 for direct float* access.
    private Tensor? _registerTokens;

    // Final AuraFlowPreFinalBlock: Linear(silu(temb)) → 2*hidden, chunk(2) → [scale, shift].
    private Tensor? _normOutLinearWeight;

    // proj_out: Linear(hidden → patch_size² * out_channels, bias=False).
    private Tensor? _projOutWeight;

    /// <summary>Creates an AuraFlow transformer from configuration.</summary>
    public AuraFlowTransformer(AuraFlowConfig config)
    {
        _config = config;

        _doubleBlocks = new AuraFlowJointBlock[config.NumDoubleBlocks];
        for (int i = 0; i < config.NumDoubleBlocks; i++)
        {
            _doubleBlocks[i] = new AuraFlowJointBlock(
                config.HiddenSize, config.NumHeads, config.HeadDim, config.MlpDim, config.QkNormEps);
        }

        _singleBlocks = new AuraFlowSingleBlock[config.NumSingleBlocks];
        for (int i = 0; i < config.NumSingleBlocks; i++)
        {
            _singleBlocks[i] = new AuraFlowSingleBlock(
                config.HiddenSize, config.NumHeads, config.HeadDim, config.MlpDim, config.QkNormEps);
        }

        _unpatchify = new Unpatchify(config.PatchSize, config.OutChannels);
    }

    /// <summary>Loads all transformer weights from named tensors using HuggingFace diffusers naming. Norm/embedding tensors that the C# code reads via raw <c>float*</c> are auto-cast to F32 at load time (cf. PHASE_3_DEVIATIONS #30/#34).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        // Patch embed: pos_embed.proj is a Linear over flattened patches.
        _patchProjWeight = weights["pos_embed.proj.weight"];
        _patchProjBias = weights.TryGetValue("pos_embed.proj.bias", out Tensor? patchBias) ? patchBias : null;

        // Learned positional embedding [1, pos_embed_max_size, inner_dim]. Read via float*; force F32.
        Tensor rawPosEmbed = weights["pos_embed.pos_embed"];
        _posEmbed = rawPosEmbed.DType != DType.F32 ? rawPosEmbed.CastTo(DType.F32) : rawPosEmbed;

        long numStored = _posEmbed.Shape[_posEmbed.Shape.Rank - 2];
        _posEmbedMaxSize = (int)numStored;
        _posEmbedAxis = (int)MathF.Round(MathF.Sqrt(numStored));
        if (_posEmbedAxis * _posEmbedAxis != numStored)
            throw new InvalidOperationException(
                $"AuraFlow pos_embed must be a square grid; got {numStored} entries (sqrt={_posEmbedAxis}).");

        // TimestepEmbedding (linear_1, linear_2 — both with biases).
        _timestepLinear1Weight = weights["time_step_proj.linear_1.weight"];
        _timestepLinear1Bias = weights["time_step_proj.linear_1.bias"];
        _timestepLinear2Weight = weights["time_step_proj.linear_2.weight"];
        _timestepLinear2Bias = weights["time_step_proj.linear_2.bias"];

        // context_embedder is bias-free.
        _contextEmbedWeight = weights["context_embedder.weight"];

        // Register tokens — [1, 8, inner_dim], read via float* during expand-and-concat.
        Tensor rawRegister = weights["register_tokens"];
        _registerTokens = rawRegister.DType != DType.F32 ? rawRegister.CastTo(DType.F32) : rawRegister;

        for (int i = 0; i < _config.NumDoubleBlocks; i++)
            _doubleBlocks[i].LoadWeights(weights, $"joint_transformer_blocks.{i}");

        for (int i = 0; i < _config.NumSingleBlocks; i++)
            _singleBlocks[i].LoadWeights(weights, $"single_transformer_blocks.{i}");

        // norm_out.linear is bias-free.
        _normOutLinearWeight = weights["norm_out.linear.weight"];

        // proj_out is bias-free.
        _projOutWeight = weights["proj_out.weight"];
    }

    /// <summary>Yields every weight tensor for GPU preloading via <c>backend.PreloadWeights</c>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_patchProjWeight is not null) yield return _patchProjWeight;
        if (_patchProjBias is not null) yield return _patchProjBias;
        if (_posEmbed is not null) yield return _posEmbed;

        if (_timestepLinear1Weight is not null) yield return _timestepLinear1Weight;
        if (_timestepLinear1Bias is not null) yield return _timestepLinear1Bias;
        if (_timestepLinear2Weight is not null) yield return _timestepLinear2Weight;
        if (_timestepLinear2Bias is not null) yield return _timestepLinear2Bias;

        if (_contextEmbedWeight is not null) yield return _contextEmbedWeight;
        if (_registerTokens is not null) yield return _registerTokens;

        for (int i = 0; i < _doubleBlocks.Length; i++)
            foreach (Tensor w in _doubleBlocks[i].EnumerateWeights()) yield return w;
        for (int i = 0; i < _singleBlocks.Length; i++)
            foreach (Tensor w in _singleBlocks[i].EnumerateWeights()) yield return w;

        if (_normOutLinearWeight is not null) yield return _normOutLinearWeight;
        if (_projOutWeight is not null) yield return _projOutWeight;
    }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="latent">Noisy latent [B, 4, H, W].</param>
    /// <param name="timestep">Scaled timestep value (matches scheduler.Timesteps[i] = sigma * 1000).</param>
    /// <param name="encoderHidden">Pile-T5-XL hidden states [B, T, joint_attention_dim=2048].</param>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor encoderHidden)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];
        int patchSize = _config.PatchSize;
        int gridH = height / patchSize;
        int gridW = width / patchSize;
        int hidden = _config.HiddenSize;

        // ── 1. Patch embed (Linear over flattened patches + learned pos_embed) ──
        Tensor imageTokens = PatchEmbedForward(backend, latent, batch, channels, height, width, gridH, gridW, hidden);
        AuraFlowDebugDump.Dump("patch_embed", imageTokens);

        // ── 2. Timestep MLP ──
        Tensor temb = ComputeTimestepEmbedding(backend, timestep, batch);
        AuraFlowDebugDump.Dump("time_text_embed", temb);

        // ── 3. Project context + prepend register tokens ──
        Tensor contextProjected = ProjectContext(backend, encoderHidden);
        Tensor textTokens = PrependRegisterTokens(contextProjected, batch);
        contextProjected.Dispose();
        AuraFlowDebugDump.Dump("context_with_registers", textTokens);

        // ── 4. Joint dual-stream blocks ──
        Tensor currentImage = imageTokens;
        Tensor currentText = textTokens;

        for (int i = 0; i < _config.NumDoubleBlocks; i++)
        {
            (Tensor newImage, Tensor newText) = _doubleBlocks[i].Forward(backend, currentImage, currentText, temb);

            currentImage.Dispose();
            currentText.Dispose();
            currentImage = newImage;
            currentText = newText;

            AuraFlowDebugDump.Dump($"joint_block_{i}_image", currentImage);
            AuraFlowDebugDump.Dump($"joint_block_{i}_text", currentText);
        }

        // ── 5. Concatenate [text, image] for single-stream stack (line 441) ──
        int textSeqLen = (int)currentText.Shape[1];
        int imgSeqLen = (int)currentImage.Shape[1];
        Tensor combined = DiTUtils.ConcatAlongSeqDim(currentText, currentImage);
        currentText.Dispose();
        currentImage.Dispose();

        // ── 6. Single-stream blocks ──
        for (int i = 0; i < _config.NumSingleBlocks; i++)
        {
            Tensor next = _singleBlocks[i].Forward(backend, combined, temb);
            combined.Dispose();
            combined = next;

            if (AuraFlowDebugDump.Enabled)
                AuraFlowDebugDump.Dump($"single_block_{i}", combined);
        }

        // ── 7. Drop the text part — keep only the image tokens (line 456) ──
        (Tensor _txtOut, Tensor imgOut) = DiTUtils.SplitAlongSeqDim(combined, textSeqLen);
        _txtOut.Dispose();
        combined.Dispose();

        // ── 8. AuraFlowPreFinalBlock: silu(temb) → Linear → [scale, shift], modulate ──
        Tensor normedOut = ApplyPreFinalBlock(backend, imgOut, temb, batch, imgSeqLen, hidden);
        imgOut.Dispose();
        AuraFlowDebugDump.Dump("norm_out", normedOut);

        // ── 9. proj_out (bias-free) ──
        int patchOutDim = patchSize * patchSize * _config.OutChannels;
        TensorShape projShape = new TensorShape(batch, imgSeqLen, patchOutDim);
        Tensor projected = new Tensor(projShape, DType.F32);
        backend.Linear(projected, normedOut, _projOutWeight!, null);
        normedOut.Dispose();
        AuraFlowDebugDump.Dump("proj_out", projected);

        // ── 10. Unpatchify → [B, out_channels, H, W] ──
        Tensor spatial = _unpatchify.Forward(projected, batch, gridH, gridW);
        projected.Dispose();
        temb.Dispose();
        AuraFlowDebugDump.DumpOutput(spatial);

        return spatial;
    }

    /// <summary>Pipeline-level debug hook: dumps the post-denoise, pre-VAE latent under <c>$AURAFLOW_DEBUG_DIR/final_latent.bin</c>.</summary>
    public static void DumpFinalLatent(Tensor latent) => AuraFlowDebugDump.Dump("final_latent", latent);

    /// <summary>Patch embed: reshape [B, C, H, W] → [B, gridH*gridW, P*P*C] then project via the patch Linear and add cropped pos_embed via <c>pe_selection_index_based_on_dim</c>.</summary>
    private Tensor PatchEmbedForward(IBackend backend, Tensor latent, int batch, int channels, int height, int width, int gridH, int gridW, int hidden)
    {
        int patchSize = _config.PatchSize;
        int numPatches = gridH * gridW;
        int patchInDim = patchSize * patchSize * channels;

        // Diffusers reshape: [B, C, H/P, P, W/P, P] → permute to [B, H/P, W/P, C, P, P] → flatten last 3 → [B, num_patches, C*P*P].
        TensorShape flatShape = new TensorShape(batch, numPatches, patchInDim);
        Tensor flat = new Tensor(flatShape, DType.F32);

        float* inPtr = (float*)latent.DataPointer;
        float* flatPtr = (float*)flat.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int gy = 0; gy < gridH; gy++)
            {
                for (int gx = 0; gx < gridW; gx++)
                {
                    int patchIdx = gy * gridW + gx;
                    int dstBase = (b * numPatches + patchIdx) * patchInDim;
                    // Layout per token: C * P * P, with C outermost (matches permute (0,2,4,1,3,5) + flatten(-3))
                    for (int c = 0; c < channels; c++)
                    {
                        for (int py = 0; py < patchSize; py++)
                        {
                            for (int px = 0; px < patchSize; px++)
                            {
                                int srcY = gy * patchSize + py;
                                int srcX = gx * patchSize + px;
                                int srcIdx = ((b * channels + c) * height + srcY) * width + srcX;
                                int dstIdx = dstBase + (c * patchSize + py) * patchSize + px;
                                flatPtr[dstIdx] = inPtr[srcIdx];
                            }
                        }
                    }
                }
            }
        }

        // Project to inner_dim via Linear.
        TensorShape outShape = new TensorShape(batch, numPatches, hidden);
        Tensor projected = new Tensor(outShape, DType.F32);
        backend.Linear(projected, flat, _patchProjWeight!, _patchProjBias);
        flat.Dispose();

        // Add cropped pos_embed (pe_selection_index_based_on_dim).
        AddPositionalEmbedding(projected, _posEmbed!, batch, gridH, gridW, hidden, _posEmbedAxis);

        return projected;
    }

    /// <summary>Adds the AuraFlow learned positional embedding cropped to a centered <c>(gridH, gridW)</c> sub-grid of the stored <c>(maxAxis, maxAxis)</c> grid. Implements <c>pe_selection_index_based_on_dim</c> from <c>AuraFlowPatchEmbed</c>.</summary>
    private static void AddPositionalEmbedding(Tensor tokens, Tensor posEmbed, int batch, int gridH, int gridW, int embedDim, int maxAxis)
    {
        float* tokPtr = (float*)tokens.DataPointer;
        float* posPtr = (float*)posEmbed.DataPointer;

        int startH = maxAxis / 2 - gridH / 2;
        int startW = maxAxis / 2 - gridW / 2;
        int numPatches = gridH * gridW;

        for (int b = 0; b < batch; b++)
        {
            for (int r = 0; r < gridH; r++)
            {
                for (int c = 0; c < gridW; c++)
                {
                    int patchIdx = r * gridW + c;
                    int posIdx = (startH + r) * maxAxis + (startW + c);
                    int tokOffset = (b * numPatches + patchIdx) * embedDim;
                    int posOffset = posIdx * embedDim;
                    for (int d = 0; d < embedDim; d++)
                        tokPtr[tokOffset + d] += posPtr[posOffset + d];
                }
            }
        }
    }

    /// <summary>Computes the AuraFlow timestep embedding: sinusoidal(timestep, 256) → Linear → SiLU → Linear → [B, inner_dim]. The scheduler timestep is already <c>sigma * 1000</c>; the diffusers <c>Timesteps(scale=1000)</c> + pipeline <c>t/1000</c> cancel to the same effective value.</summary>
    private Tensor ComputeTimestepEmbedding(IBackend backend, float timestep, int batch)
    {
        int hidden = _config.HiddenSize;

        TensorShape sinShape = new TensorShape(batch, 256);
        Tensor sinEmbed = new Tensor(sinShape, DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinEmbed, timestep, batch, embDim: 256);

        TensorShape hidShape = new TensorShape(batch, hidden);

        Tensor t1 = new Tensor(hidShape, DType.F32);
        backend.Linear(t1, sinEmbed, _timestepLinear1Weight!, _timestepLinear1Bias);
        sinEmbed.Dispose();

        Tensor t1Activated = new Tensor(hidShape, DType.F32);
        backend.Silu(t1Activated, t1);
        t1.Dispose();

        Tensor temb = new Tensor(hidShape, DType.F32);
        backend.Linear(temb, t1Activated, _timestepLinear2Weight!, _timestepLinear2Bias);
        t1Activated.Dispose();

        return temb;
    }

    /// <summary>Projects context [B, T, joint_attention_dim] → [B, T, caption_projection_dim] via the bias-free <c>context_embedder</c>.</summary>
    private Tensor ProjectContext(IBackend backend, Tensor context)
    {
        int batch = (int)context.Shape[0];
        int seqLen = (int)context.Shape[1];

        TensorShape outShape = new TensorShape(batch, seqLen, _config.CaptionProjectionDim);
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Linear(output, context, _contextEmbedWeight!, null);

        return output;
    }

    /// <summary>Prepends the learned <c>register_tokens</c> [1, 8, hidden] to the per-batch context tokens. Implements <c>register_tokens.repeat(B, 1, 1)</c> + <c>cat([reg, ctx], dim=1)</c>.</summary>
    private Tensor PrependRegisterTokens(Tensor context, int batch)
    {
        int seqLen = (int)context.Shape[1];
        int hidden = (int)context.Shape[2];
        int numRegister = _config.NumRegisterTokens;
        int outSeq = numRegister + seqLen;

        TensorShape outShape = new TensorShape(batch, outSeq, hidden);
        Tensor output = new Tensor(outShape, DType.F32);

        float* regPtr = (float*)_registerTokens!.DataPointer;
        float* ctxPtr = (float*)context.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        long regBytes = (long)numRegister * hidden * sizeof(float);
        long ctxBytes = (long)seqLen * hidden * sizeof(float);

        for (int b = 0; b < batch; b++)
        {
            int outBase = b * outSeq * hidden;
            // Register tokens are stored once [1, 8, hidden] and broadcast across batch.
            Buffer.MemoryCopy(regPtr, outPtr + outBase, regBytes, regBytes);
            Buffer.MemoryCopy(ctxPtr + b * seqLen * hidden, outPtr + outBase + numRegister * hidden, ctxBytes, ctxBytes);
        }

        return output;
    }

    /// <summary>Applies the AuraFlowPreFinalBlock: <c>emb = silu(temb) → Linear; scale, shift = chunk(emb, 2)</c>; <c>x = x * (1 + scale) + shift</c>. Note the chunk order is <c>[scale, shift]</c>, not the SD3 <c>[shift, scale]</c>.</summary>
    private Tensor ApplyPreFinalBlock(IBackend backend, Tensor hidden, Tensor temb, int batch, int seqLen, int dim)
    {
        TensorShape tembShape = new TensorShape(batch, dim);
        Tensor activated = new Tensor(tembShape, DType.F32);
        backend.Silu(activated, temb);

        TensorShape modShape = new TensorShape(batch, dim * 2);
        Tensor mod = new Tensor(modShape, DType.F32);
        backend.Linear(mod, activated, _normOutLinearWeight!, null);
        activated.Dispose();

        // AuraFlow chunk order: [scale, shift] (lines 141-142 of auraflow_transformer_2d.py).
        TensorShape outShape = new TensorShape(batch, seqLen, dim);
        Tensor output = new Tensor(outShape, DType.F32);

        float* hidPtr = (float*)hidden.DataPointer;
        float* modPtr = (float*)mod.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int modBase = b * dim * 2;
            for (int s = 0; s < seqLen; s++)
            {
                int hidOffset = (b * seqLen + s) * dim;
                for (int d = 0; d < dim; d++)
                {
                    float scale = modPtr[modBase + d];
                    float shift = modPtr[modBase + dim + d];
                    outPtr[hidOffset + d] = hidPtr[hidOffset + d] * (1.0f + scale) + shift;
                }
            }
        }

        mod.Dispose();
        return output;
    }

    /// <summary>Releases all tensor references held by this transformer.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _patchProjWeight = null;
            _patchProjBias = null;
            _posEmbed = null;
            _timestepLinear1Weight = null;
            _timestepLinear1Bias = null;
            _timestepLinear2Weight = null;
            _timestepLinear2Bias = null;
            _contextEmbedWeight = null;
            _registerTokens = null;
            _normOutLinearWeight = null;
            _projOutWeight = null;
        }
        GC.SuppressFinalize(this);
    }
}
