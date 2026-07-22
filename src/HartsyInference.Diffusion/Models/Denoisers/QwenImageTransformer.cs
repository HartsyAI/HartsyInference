using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Qwen-Image MMDiT transformer (<c>QwenImageTransformer2DModel</c>). Processes packed image patch tokens and Qwen2.5-VL text embeddings through 60 dual-stream transformer blocks with QK-norm, AdaLN-Zero modulation, and 3-axis (frame, height, width) RoPE applied separately to each stream before joint attention. Top-level layout follows <c>diffusers/models/transformers/transformer_qwenimage.py</c>: <c>img_in</c> Linear → <c>txt_norm</c> RMSNorm → <c>txt_in</c> Linear → <c>time_text_embed</c> sinusoidal+MLP → 60 × <see cref="QwenImageBlock"/> → <c>norm_out</c> AdaLN-continuous → <c>proj_out</c> Linear. Outputs predicted velocity in packed token form <c>[B, imgSeqLen, patch_size² * out_channels]</c>; pipeline unpacks back to <c>[B, C, H, W]</c>.</summary>
public sealed unsafe class QwenImageTransformer : IDisposable
{
    private readonly QwenImageConfig _config;
    private readonly QwenImageBlock[] _blocks;
    private readonly QwenImageRope _rope;

    /// <summary>True when this instance runs the audited F16 block loop (HARTSY_DIT_F16) with the exact
    /// <see cref="ChromaF16.ResidualDamp"/> residual damp — the Chroma/Flux recipe: every branch input passes
    /// a no-affine LayerNorm and the final AdaLN-continuous norm cancels the factor before proj_out.</summary>
    private bool _f16Mode;

    /// <summary>Qwen-Image's residual stream is an outlier among DiTs: massive-activation channels push it to
    /// ±10M by mid-depth (measured block-input absmax, 60-block V1) — Flux/Chroma stay under ~65k, which is why
    /// their shared <see cref="ChromaF16.ResidualDamp"/> (1/32) overflowed F16 here after ONE block. 1/512
    /// (2^-9, exact) parks the plateau at ~±20k with headroom; the no-affine LayerNorms still cancel it.</summary>
    private const float QwenResidualDamp = 1.0f / 8192.0f;
    private int _disposed;

    private Tensor? _imgInWeight, _imgInBias;

    private Tensor? _txtNormWeight;

    private Tensor? _txtInWeight, _txtInBias;

    private Tensor? _timestepLinear1Weight, _timestepLinear1Bias;
    private Tensor? _timestepLinear2Weight, _timestepLinear2Bias;

    private Tensor? _normOutLinearWeight, _normOutLinearBias;
    private Tensor? _projOutWeight, _projOutBias;

    /// <summary>Creates a Qwen-Image transformer from configuration.</summary>
    public QwenImageTransformer(QwenImageConfig config)
    {
        _config = config;
        int mlpDim = (int)(config.HiddenSize * config.MlpRatio);

        _blocks = new QwenImageBlock[config.Depth];
        for (int i = 0; i < config.Depth; i++)
        {
            _blocks[i] = new QwenImageBlock(
                config.HiddenSize,
                config.NumHeads,
                config.HeadDim,
                mlpDim,
                config.QkNormEps);
        }

        _rope = new QwenImageRope(theta: config.RopeTheta);
    }

    /// <summary>Loads all transformer weights from named tensors using diffusers naming.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        // F16 opt-in DISABLED for Qwen-Image (2026-07-10 finding): the residual stream's massive-activation
        // channels (±10M plateau — 150× past F16 even before growth) forced a 1/2048 damp, which exposed a
        // second bug (LayerNorm eps breaks damp scale-invariance; fixed below via eps·damp²) — but even with
        // both fixes and single-forward exactness (corr 0.99998), the SECOND denoise step's forward NaNs
        // deterministically (not a race — reproduces under CUDA_LAUNCH_BLOCKING). The reference stacks run
        // this model in BF16 for exactly this range reason; a BF16 activation path is the correct future
        // lever. Until then Qwen stays F32 (still beats ComfyUI on t2i; edit ~1.05× of Comfy).
        _f16Mode = false;
        float branchDamp = _f16Mode ? QwenResidualDamp : 1.0f;

        _imgInWeight = weights["img_in.weight"];
        _imgInBias = weights["img_in.bias"];

        _txtNormWeight = CastToF32IfNeeded(weights["txt_norm.weight"]);

        _txtInWeight = weights["txt_in.weight"];
        _txtInBias = weights["txt_in.bias"];
        if (_f16Mode)
        {
            // Enter the damped-residual regime at the embedders (see ChromaF16): both token streams start at
            // damp scale; the block-output damping keeps them there; the final no-affine LayerNorm cancels
            // the factor exactly. Weight damp rides the GEMM alpha (dequantized-GGUF cuBLAS path included).
            _imgInWeight.Fp8ScaleFactor *= QwenResidualDamp;
            _imgInBias = ChromaF16.DampBias(_imgInBias!, QwenResidualDamp);
            _txtInWeight.Fp8ScaleFactor *= QwenResidualDamp;
            _txtInBias = ChromaF16.DampBias(_txtInBias!, QwenResidualDamp);
            Logs.Info($"[QwenImage] F16 block loop active (residual damp 1/{1.0f / QwenResidualDamp:F0})");
        }

        _timestepLinear1Weight = weights["time_text_embed.timestep_embedder.linear_1.weight"];
        _timestepLinear1Bias = weights["time_text_embed.timestep_embedder.linear_1.bias"];
        _timestepLinear2Weight = weights["time_text_embed.timestep_embedder.linear_2.weight"];
        _timestepLinear2Bias = weights["time_text_embed.timestep_embedder.linear_2.bias"];

        for (int i = 0; i < _config.Depth; i++)
            _blocks[i].LoadWeights(weights, $"transformer_blocks.{i}", branchDamp);

        _normOutLinearWeight = weights["norm_out.linear.weight"];
        _normOutLinearBias = weights["norm_out.linear.bias"];
        _projOutWeight = weights["proj_out.weight"];
        _projOutBias = weights["proj_out.bias"];
    }

    /// <summary>Yields every weight tensor for GPU preloading via <see cref="IBackend.PreloadWeights"/>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_imgInWeight is not null) yield return _imgInWeight;
        if (_imgInBias is not null) yield return _imgInBias;
        if (_txtNormWeight is not null) yield return _txtNormWeight;
        if (_txtInWeight is not null) yield return _txtInWeight;
        if (_txtInBias is not null) yield return _txtInBias;
        if (_timestepLinear1Weight is not null) yield return _timestepLinear1Weight;
        if (_timestepLinear1Bias is not null) yield return _timestepLinear1Bias;
        if (_timestepLinear2Weight is not null) yield return _timestepLinear2Weight;
        if (_timestepLinear2Bias is not null) yield return _timestepLinear2Bias;

        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;

        if (_normOutLinearWeight is not null) yield return _normOutLinearWeight;
        if (_normOutLinearBias is not null) yield return _normOutLinearBias;
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
    }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="packedLatent">Packed latent tokens [B, imgSeqLen, patch_size² * in_channels]. For
    /// Qwen-Image-Edit the caller appends the packed reference latent AFTER the noise tokens
    /// (<c>[noise ; ref]</c> along the sequence dim) and passes the ref grid via
    /// <paramref name="refHPacked"/>/<paramref name="refWPacked"/>.</param>
    /// <param name="encoderHidden">Qwen2.5-VL encoder hidden states [B, txtSeqLen, encoderDim].</param>
    /// <param name="timestep">Normalized timestep in [0, 1] (diffusers passes <c>t / 1000</c>).</param>
    /// <param name="hPacked">Packed-grid height (<c>latent_h / patch_size</c>) of the MAIN (noise) latent.</param>
    /// <param name="wPacked">Packed-grid width (<c>latent_w / patch_size</c>) of the MAIN (noise) latent.</param>
    /// <param name="refGrids">Packed grids of the trailing Qwen-Image-Edit reference-latent token sections,
    /// in sequence order (null/empty = none). Ref section <c>i</c> runs through every block as image tokens
    /// with frame-axis <c>i+1</c> RoPE (ComfyUI <c>qwen_image/model.py</c> ref "index" method) and is DROPPED
    /// from the output — the returned velocity covers only the main <c>hPacked·wPacked</c> tokens (upstream
    /// <c>hidden_states[:, :num_embeds]</c>).</param>
    /// <param name="refTimestepZero">Qwen-Image-Edit-2511 (<c>index_timestep_zero</c>): modulate/gate the
    /// reference-latent rows with the t=0 modulation in every block (ComfyUI <c>timestep_zero_index</c>).
    /// The 2509 checkpoints use plain "index" (false). Ignored when no ref tokens are present.</param>
    /// <param name="stepCache">Optional across-step First-Block cache (one instance PER CFG stream). Block 0
    /// always runs; on a gate hit blocks 1..N−1 are reconstructed from the previous step's residual instead of
    /// computed. Null (the default) is byte-identical to the uncached forward.</param>
    public Tensor Forward(IBackend backend, Tensor packedLatent, Tensor encoderHidden, float timestep,
        int hPacked, int wPacked, (int H, int W)[]? refGrids = null, bool refTimestepZero = false,
        DeviceFeatureCache? stepCache = null)
    {
        int batch = (int)packedLatent.Shape[0];
        int imgSeqLen = (int)packedLatent.Shape[1];
        int txtSeqLen = (int)encoderHidden.Shape[1];
        int hidden = _config.HiddenSize;
        refGrids ??= [];
        int refSeqLen = 0;
        foreach ((int rh, int rw) in refGrids) refSeqLen += rh * rw;
        int mainSeqLen = imgSeqLen - refSeqLen;
        if (refSeqLen > 0 && mainSeqLen != hPacked * wPacked)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                $"packedLatent seq {imgSeqLen} must equal main ({hPacked}x{wPacked}) + {refSeqLen} ref tokens.");
        // The output slice below (and the block's seq-dim [txt|img] split) assume contiguous batch-1 rows.
        if (refSeqLen > 0 && batch != 1)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                "Reference-latent (edit) tokens require batch size 1; run CFG as two batch-1 passes.");

        TensorShape imgTokShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor imgTokens = new Tensor(imgTokShape, DType.F32);
        backend.Linear(imgTokens, packedLatent, _imgInWeight!, _imgInBias);
        QwenImageDebugDump.Dump("img_in", imgTokens);

        TensorShape txtNormShape = encoderHidden.Shape;
        Tensor txtNormed = new Tensor(txtNormShape, DType.F32);
        backend.RmsNorm(txtNormed, encoderHidden, _txtNormWeight!, 1e-6f);

        TensorShape txtTokShape = new TensorShape(batch, txtSeqLen, hidden);
        Tensor txtTokens = new Tensor(txtTokShape, DType.F32);
        backend.Linear(txtTokens, txtNormed, _txtInWeight!, _txtInBias);
        txtNormed.Dispose();
        QwenImageDebugDump.Dump("txt_in", txtTokens);

        // F16 block loop (HARTSY_DIT_F16, B=1): one cast per stream before the loop — every block
        // activation follows, and all 60 SDPAs run zero-cast cuDNN F16. Streams already ride at
        // ResidualDamp scale from the damped embedders. Cast back to F32 after the loop for the final
        // norm (which cancels the damp) + proj_out.
        bool f16Loop = _f16Mode && batch == 1;
        if (f16Loop)
        {
            Tensor imgF16 = new Tensor(imgTokShape, DType.F16);
            backend.CastToF16(imgF16, imgTokens);
            imgTokens.Dispose();
            imgTokens = imgF16;
            Tensor txtF16 = new Tensor(txtTokShape, DType.F16);
            backend.CastToF16(txtF16, txtTokens);
            txtTokens.Dispose();
            txtTokens = txtF16;
        }

        Tensor temb = ComputeTimestepEmbedding(backend, timestep, batch);
        QwenImageDebugDump.Dump("time_text_embed", temb);

        // 2511 timestep-zero ref method: a second temb at t=0 drives the ref-row modulation in every block
        // (identical to upstream's batch-2 `cat([timestep, timestep*0])` — separate embedding call, same math).
        Tensor? tembZero = refSeqLen > 0 && refTimestepZero
            ? ComputeTimestepEmbedding(backend, 0.0f, batch)
            : null;

        int txtPositionStart = QwenImageRope.ComputeTextPositionStart(hPacked, wPacked);

        Tensor currentImg = imgTokens;
        Tensor currentTxt = txtTokens;

        // Across-step First-Block cache: block 0 always runs; its img stream is the gate indicator. On a hit,
        // blocks 1..N−1 are skipped and the final hidden state is block0 + the previous full compute's residual
        // (device Add). On a miss the anchor (block-0 output) survives the loop so the fresh residual can be
        // stored. Null stepCache leaves this loop byte-identical to the original.
        Tensor? cacheAnchor = null;
        bool cacheHit = false;
        int startBlock = 0;
        if (stepCache is not null && _config.Depth > 1)
        {
            (Tensor img0, Tensor txt0) = _blocks[0].Forward(
                backend, currentImg, currentTxt, temb, _rope,
                hPacked, wPacked, txtPositionStart, refGrids, tembZero, mainSeqLen);
            currentImg.Dispose();
            currentTxt.Dispose();
            currentImg = img0;
            currentTxt = txt0;
            QwenImageDebugDump.Dump("block_0_image", currentImg);
            QwenImageDebugDump.Dump("block_0_text", currentTxt);

            startBlock = 1;
            cacheHit = !stepCache.ShouldCompute(backend, currentImg);
            if (cacheHit)
            {
                Tensor reconstructed = stepCache.ApplyResidual(backend, currentImg);
                currentImg.Dispose();
                currentImg = reconstructed;
                startBlock = _config.Depth;
            }
            else
            {
                cacheAnchor = currentImg;
            }
        }

        for (int i = startBlock; i < _config.Depth; i++)
        {
            (Tensor newImg, Tensor newTxt) = _blocks[i].Forward(
                backend, currentImg, currentTxt, temb, _rope,
                hPacked, wPacked, txtPositionStart, refGrids, tembZero, mainSeqLen);

            if (currentImg != cacheAnchor) currentImg.Dispose();
            currentTxt.Dispose();

            currentImg = newImg;
            currentTxt = newTxt;

            QwenImageDebugDump.Dump($"block_{i}_image", currentImg);
            QwenImageDebugDump.Dump($"block_{i}_text", currentTxt);
        }

        if (cacheAnchor is not null)
        {
            stepCache!.StoreResidual(backend, cacheAnchor, currentImg);
            cacheAnchor.Dispose();
        }

        currentTxt.Dispose();

        if (currentImg.DType == DType.F16)
        {
            // Back to F32 for the final norm + proj_out (velocity precision across Euler steps).
            Tensor imgF32 = new Tensor(imgTokShape, DType.F32);
            backend.CastToF32(imgF32, currentImg);
            currentImg.Dispose();
            currentImg = imgF32;
        }

        Tensor output = ApplyFinalLayer(backend, currentImg, temb, batch, imgSeqLen);
        QwenImageDebugDump.Dump("proj_out", output);
        currentImg.Dispose();
        temb.Dispose();
        tembZero?.Dispose();

        // Qwen-Image-Edit: drop the reference-latent rows — only the main noise tokens carry velocity
        // (upstream `hidden_states[:, :num_embeds]`, applied AFTER norm_out/proj_out; per-token final
        // layer means slicing after is numerically identical to slicing before).
        if (refSeqLen > 0)
        {
            int outDim = _config.PatchSize * _config.PatchSize * _config.InChannels;
            Tensor mainOnly = new Tensor(new TensorShape(batch, mainSeqLen, outDim), DType.F32);
            backend.SliceRows(mainOnly, output, 0);
            output.Dispose();
            output = mainOnly;
        }

        QwenImageDebugDump.DumpOutput(output);
        return output;
    }

    private Tensor ComputeTimestepEmbedding(IBackend backend, float timestep, int batch)
    {
        int hidden = _config.HiddenSize;
        float scaledTimestep = timestep * 1000.0f;

        TensorShape sinShape = new TensorShape(batch, 256);
        Tensor sinEmbed = new Tensor(sinShape, DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinEmbed, scaledTimestep, batch, embDim: 256);

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

    /// <summary>Final AdaLN-continuous: <c>SiLU(temb) → Linear → [shift, scale]</c> then unparameterized LayerNorm + modulate + <c>proj_out</c>. Diffusers' AdaLayerNormContinuous chunks <c>[shift, scale]</c> in that order — matches the Linear output layout used by SD3's final layer.</summary>
    private Tensor ApplyFinalLayer(IBackend backend, Tensor hidden, Tensor temb, int batch, int seqLen)
    {
        int dim = _config.HiddenSize;
        int outDim = _config.PatchSize * _config.PatchSize * _config.InChannels;
        TensorShape hidShape = new TensorShape(batch, seqLen, dim);

        TensorShape tembShape = new TensorShape(batch, dim);
        Tensor activated = new Tensor(tembShape, DType.F32);
        backend.Silu(activated, temb);

        TensorShape modParamShape = new TensorShape(batch, dim * 2);
        Tensor modParams = new Tensor(modParamShape, DType.F32);
        backend.Linear(modParams, activated, _normOutLinearWeight!, _normOutLinearBias);
        activated.Dispose();

        Tensor normed = new Tensor(hidShape, DType.F32);
        DiTUtils.LayerNormNoAffine(normed, hidden, batch, seqLen, dim);

        Tensor modulated = new Tensor(hidShape, DType.F32);
        float* modPtr = (float*)modParams.DataPointer;
        float* normPtr = (float*)normed.DataPointer;
        float* outModPtr = (float*)modulated.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int modBase = b * dim * 2;
            for (int s = 0; s < seqLen; s++)
            {
                int vecOffset = (b * seqLen + s) * dim;
                for (int d = 0; d < dim; d++)
                {
                    // AdaLayerNormContinuous (Qwen-Image norm_out / ComfyUI LastLayer): `scale, shift = chunk(emb, 2)`
                    // — SCALE is the first half, SHIFT the second. (NOT the AdaLayerNormZero [shift,scale] order the
                    // per-block modulation uses.) Was swapped → distorted final velocity.
                    float scale = modPtr[modBase + d];
                    float shift = modPtr[modBase + dim + d];
                    outModPtr[vecOffset + d] = normPtr[vecOffset + d] * (1.0f + scale) + shift;
                }
            }
        }
        normed.Dispose();
        modParams.Dispose();
        QwenImageDebugDump.Dump("norm_out", modulated);

        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor projected = new Tensor(outShape, DType.F32);
        backend.Linear(projected, modulated, _projOutWeight!, _projOutBias);
        modulated.Dispose();

        return projected;
    }

    /// <summary>Pipeline-level debug hook: dumps the post-denoise, pre-VAE latent under <c>$QWEN_IMAGE_DEBUG_DIR/final_latent.bin</c> when the env var is set. Used by the layer-by-layer diff harness to capture pipeline state.</summary>
    public static void DumpFinalLatent(Tensor latent) => QwenImageDebugDump.Dump("final_latent", latent);

    private static Tensor CastToF32IfNeeded(Tensor t) =>
        t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    /// <summary>Releases all tensor references.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _imgInWeight = null; _imgInBias = null;
            _txtNormWeight = null;
            _txtInWeight = null; _txtInBias = null;
            _timestepLinear1Weight = null; _timestepLinear1Bias = null;
            _timestepLinear2Weight = null; _timestepLinear2Bias = null;
            _normOutLinearWeight = null; _normOutLinearBias = null;
            _projOutWeight = null; _projOutBias = null;
        }
        GC.SuppressFinalize(this);
    }
}
