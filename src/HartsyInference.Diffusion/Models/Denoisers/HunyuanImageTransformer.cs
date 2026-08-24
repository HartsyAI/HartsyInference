using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Hunyuan Image 2.1 MMDiT transformer (<c>HunyuanImageTransformer2DModel</c>). Processes 64-channel latents with a 32× VAE downscale and unit patches (<c>patch_size=1</c>) through 20 dual-stream <see cref="HunyuanImageBlock"/>s and 40 single-stream <see cref="HunyuanImageSingleBlock"/>s with shared image-only RoPE. Top-level layout follows <c>diffusers/models/transformers/transformer_hunyuanimage.py</c>: <c>x_embedder</c> 1x1 patchify conv → <c>context_embedder</c> (Hunyuan token refiner: pooled prompt + timestep → AdaNorm-gated self-attn × N) → <c>time_guidance_embed</c> sinusoidal+MLP (timestep + optional distilled-guidance) → 20 × <see cref="HunyuanImageBlock"/> → 40 × <see cref="HunyuanImageSingleBlock"/> → <c>norm_out</c> AdaLN-continuous → <c>proj_out</c> Linear(hidden, patch² * out_channels). Outputs predicted velocity in patched form <c>[B, postPatchH * postPatchW, patch² * out_channels]</c>; the pipeline unpatchifies back to <c>[B, 64, H, W]</c>. ByT5 secondary text encoder is supported when <see cref="HunyuanImageConfig.TextEmbedDim2"/> is non-null and a <c>encoder_hidden_states_2</c> tensor is supplied to <see cref="Forward"/>.</summary>
public sealed unsafe class HunyuanImageTransformer : IDisposable
{
    private readonly HunyuanImageConfig _config;
    private readonly HunyuanImageBlock[] _doubleBlocks;
    private readonly HunyuanImageSingleBlock[] _singleBlocks;
    private readonly HunyuanImageRope _rope;
    private readonly HunyuanImageTokenRefiner _contextEmbedder;
    private readonly HunyuanImageByT5Projection? _contextEmbedder2;
    private int _disposed;

    private Tensor? _xEmbedWeight, _xEmbedBias;

    private Tensor? _timeProjLinear1Weight, _timeProjLinear1Bias;
    private Tensor? _timeProjLinear2Weight, _timeProjLinear2Bias;

    private Tensor? _guidanceLinear1Weight, _guidanceLinear1Bias;
    private Tensor? _guidanceLinear2Weight, _guidanceLinear2Bias;

    private Tensor? _normOutLinearWeight, _normOutLinearBias;
    private Tensor? _projOutWeight, _projOutBias;

    public HunyuanImageTransformer(HunyuanImageConfig config)
    {
        _config = config;
        int mlpDim = (int)(config.HiddenSize * config.MlpRatio);

        _doubleBlocks = new HunyuanImageBlock[config.NumDoubleBlocks];
        for (int i = 0; i < config.NumDoubleBlocks; i++)
        {
            _doubleBlocks[i] = new HunyuanImageBlock(
                config.HiddenSize,
                config.NumHeads,
                config.HeadDim,
                mlpDim,
                config.QkNormEps);
        }

        _singleBlocks = new HunyuanImageSingleBlock[config.NumSingleBlocks];
        for (int i = 0; i < config.NumSingleBlocks; i++)
        {
            _singleBlocks[i] = new HunyuanImageSingleBlock(
                config.HiddenSize,
                config.NumHeads,
                config.HeadDim,
                mlpDim,
                config.QkNormEps);
        }

        _rope = new HunyuanImageRope(config.RopeAxesDim, config.RopeTheta);

        _contextEmbedder = new HunyuanImageTokenRefiner(
            config.TextEmbedDim,
            config.HiddenSize,
            config.NumHeads,
            config.HeadDim,
            config.NumRefinerLayers,
            mlpRatio: config.MlpRatio,
            qkNormEps: config.QkNormEps);

        _contextEmbedder2 = config.TextEmbedDim2 is int dim2
            ? new HunyuanImageByT5Projection(dim2, intermediateDim: 2048, outFeatures: config.HiddenSize) : null;
    }

    /// <summary>Loads all transformer weights from named tensors using diffusers naming.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _xEmbedWeight = weights["x_embedder.proj.weight"];
        _xEmbedBias = weights["x_embedder.proj.bias"];

        _timeProjLinear1Weight = weights["time_guidance_embed.timestep_embedder.linear_1.weight"];
        _timeProjLinear1Bias = weights["time_guidance_embed.timestep_embedder.linear_1.bias"];
        _timeProjLinear2Weight = weights["time_guidance_embed.timestep_embedder.linear_2.weight"];
        _timeProjLinear2Bias = weights["time_guidance_embed.timestep_embedder.linear_2.bias"];

        if (_config.GuidanceEmbed)
        {
            _guidanceLinear1Weight = weights["time_guidance_embed.guidance_embedder.linear_1.weight"];
            _guidanceLinear1Bias = weights["time_guidance_embed.guidance_embedder.linear_1.bias"];
            _guidanceLinear2Weight = weights["time_guidance_embed.guidance_embedder.linear_2.weight"];
            _guidanceLinear2Bias = weights["time_guidance_embed.guidance_embedder.linear_2.bias"];
        }

        _contextEmbedder.LoadWeights(weights, "context_embedder");
        _contextEmbedder2?.LoadWeights(weights, "context_embedder_2");

        for (int i = 0; i < _config.NumDoubleBlocks; i++)
            _doubleBlocks[i].LoadWeights(weights, $"transformer_blocks.{i}");

        for (int i = 0; i < _config.NumSingleBlocks; i++)
            _singleBlocks[i].LoadWeights(weights, $"single_transformer_blocks.{i}");

        _normOutLinearWeight = weights["norm_out.linear.weight"];
        _normOutLinearBias = weights["norm_out.linear.bias"];
        _projOutWeight = weights["proj_out.weight"];
        _projOutBias = weights["proj_out.bias"];
    }

    /// <summary>Yields every weight tensor for GPU preloading via <see cref="IBackend.PreloadWeights"/>. Equivalent to <see cref="EnumerateSharedWeights"/> followed by every streamable block's weights — the eager all-at-once preload used when the whole DiT fits resident.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in EnumerateSharedWeights()) yield return w;
        for (int i = 0; i < BlockCount; i++)
            foreach (Tensor w in GetBlock(i).EnumerateWeights()) yield return w;
    }

    /// <summary>Yields the always-resident weights — <c>x_embedder</c>, the timestep/guidance MLPs, the context embedder(s), and the final AdaLN + <c>proj_out</c>. These are touched on every forward regardless of which block is executing, so the streaming controller never manages them; callers on the streaming path preload these eagerly. Sizeable here (the Hunyuan token refiner is 2 full transformer layers at <see cref="HunyuanImageConfig.HiddenSize"/>), so a streaming budget must count them explicitly rather than folding them into a fudge factor.</summary>
    public IEnumerable<Tensor> EnumerateSharedWeights()
    {
        if (_xEmbedWeight is not null) yield return _xEmbedWeight;
        if (_xEmbedBias is not null) yield return _xEmbedBias;
        if (_timeProjLinear1Weight is not null) yield return _timeProjLinear1Weight;
        if (_timeProjLinear1Bias is not null) yield return _timeProjLinear1Bias;
        if (_timeProjLinear2Weight is not null) yield return _timeProjLinear2Weight;
        if (_timeProjLinear2Bias is not null) yield return _timeProjLinear2Bias;
        if (_guidanceLinear1Weight is not null) yield return _guidanceLinear1Weight;
        if (_guidanceLinear1Bias is not null) yield return _guidanceLinear1Bias;
        if (_guidanceLinear2Weight is not null) yield return _guidanceLinear2Weight;
        if (_guidanceLinear2Bias is not null) yield return _guidanceLinear2Bias;

        foreach (Tensor w in _contextEmbedder.EnumerateWeights()) yield return w;
        if (_contextEmbedder2 is not null)
            foreach (Tensor w in _contextEmbedder2.EnumerateWeights()) yield return w;

        if (_normOutLinearWeight is not null) yield return _normOutLinearWeight;
        if (_normOutLinearBias is not null) yield return _normOutLinearBias;
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
    }

    /// <summary>Weights of flat blocks <c>[startBlock, endBlock)</c> only — the asymmetric-preload primitive for DiT sharding: backend A preloads <see cref="EnumerateSharedWeights"/> + its range, backend B ONLY its range. Never preload <see cref="EnumerateWeights"/> on both backends — that replicates instead of pooling.</summary>
    public IEnumerable<Tensor> EnumerateBlockRangeWeights(int startBlock, int endBlock)
    {
        for (int i = startBlock; i < endBlock; i++)
            foreach (Tensor w in GetBlock(i).EnumerateWeights()) yield return w;
    }

    /// <summary>Number of streamable blocks: the 20 double-stream blocks occupy <c>[0, NumDoubleBlocks)</c>, the 40 single-stream blocks <c>[NumDoubleBlocks, BlockCount)</c>.</summary>
    public int BlockCount => _doubleBlocks.Length + _singleBlocks.Length;

    /// <summary>Streamable block at global index <paramref name="idx"/> (double blocks first, then single). Both block types implement <see cref="IStreamingBlock"/> directly, so this returns the live instance — its <c>EnumerateWeights</c> hands back the same tensor references every call, which the controller's residency tracking requires.</summary>
    public IStreamingBlock GetBlock(int idx)
    {
        if (idx < 0 || idx >= BlockCount) throw new ArgumentOutOfRangeException(nameof(idx));
        return idx < _doubleBlocks.Length ? _doubleBlocks[idx] : _singleBlocks[idx - _doubleBlocks.Length];
    }

    /// <summary>Optional hook invoked immediately before each block's forward pass, with the same global index <see cref="GetBlock"/> uses. Pipelines plug a <see cref="BlockStreamingController"/> here to drive prefetch/eviction so the 17B DiT fits a 12 GB card. Null (the default) leaves the transformer behaving exactly as before — the caller must have every weight resident. This transformer has no captured-CUDA-graph fast path, so there is nothing to guard against here; if one is ever added it must be disabled whenever this hook is non-null (a graph bakes weight device pointers and streaming re-points them every forward).</summary>
    public Action<int>? BeforeBlockForward { get; set; }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="patchedLatent">Patched latent tokens <c>[B, postPatchH * postPatchW, patch² * in_channels]</c>. With <c>patch_size=1</c>, <c>postPatchH=H</c>, <c>postPatchW=W</c>, and the per-token feature dim equals <c>in_channels</c>.</param>
    /// <param name="encoderHidden">Primary (MLLM/Qwen2.5-VL) text embeddings <c>[B, txtSeqLen, TextEmbedDim]</c>.</param>
    /// <param name="encoderHidden2">Optional ByT5 secondary text embeddings <c>[B, txtSeqLen2, TextEmbedDim2]</c>; ignored when null or when <see cref="HunyuanImageConfig.TextEmbedDim2"/> is null.</param>
    /// <param name="timestep">Timestep value (already in the same scale as the diffusers reference: 0..1000 for the standard 1000-step schedule).</param>
    /// <param name="guidanceScale">Embedded distilled-guidance scale (ignored when <see cref="HunyuanImageConfig.GuidanceEmbed"/> is false).</param>
    /// <param name="postPatchH">Image grid height (<c>latent_h / patch_size</c>).</param>
    /// <param name="postPatchW">Image grid width (<c>latent_w / patch_size</c>).</param>
    public Tensor Forward(IBackend backend, Tensor patchedLatent, Tensor encoderHidden, Tensor? encoderHidden2,
        float timestep, float guidanceScale, int postPatchH, int postPatchW)
    {
        int batch = (int)patchedLatent.Shape[0];
        int imgSeqLen = (int)patchedLatent.Shape[1];

        (Tensor currentImg, Tensor currentTxt, Tensor temb) = ForwardEmbedIn(
            backend, patchedLatent, encoderHidden, encoderHidden2, timestep, guidanceScale);

        ForwardBlocksRange(backend, ref currentImg, ref currentTxt, temb, postPatchH, postPatchW, 0, BlockCount);

        currentTxt.Dispose();

        Tensor output = ForwardHeadOut(backend, currentImg, temb, batch, imgSeqLen);
        temb.Dispose();
        return output;
    }

    /// <summary>DiT-sharded forward: flat blocks <c>[0, splitBlock)</c> on <paramref name="backendA"/> (which also owns the shared embed/refiner/head weights), <c>[splitBlock, BlockCount)</c> on <paramref name="backendB"/>, with the img+txt streams and temb handed across via <see cref="IBackend.CopyFromPeer"/> and the img stream handed back for the head. VRAM pooling, not latency — the two backends run sequentially. Both regions carry the (img, txt) pair (single blocks concat/split it internally per block), so the handoff is uniform wherever the split lands — inside the doubles, inside the singles, or on the region boundary. Any ByT5 tokens were concatenated into the txt stream by the embed-in on A, so they ride the txt handoff. Exclusions: no block streaming (<see cref="BeforeBlockForward"/> must be null — the sharding preload owns block residency); no step-graph exists here; the loop is pinned F32 (this arch never opts into HARTSY_DIT_F16 — see the range helper's comment), matching the unsharded forward exactly. Callers preload <see cref="EnumerateSharedWeights"/> + <see cref="EnumerateBlockRangeWeights"/>(0, split) on A and <see cref="EnumerateBlockRangeWeights"/>(split, BlockCount) on B.</summary>
    public Tensor ForwardSharded(IBackend backendA, IBackend backendB, Tensor patchedLatent, Tensor encoderHidden,
        Tensor? encoderHidden2, float timestep, float guidanceScale, int postPatchH, int postPatchW, int splitBlock)
    {
        if (splitBlock <= 0 || splitBlock >= BlockCount)
            throw new ArgumentOutOfRangeException(nameof(splitBlock),
                $"splitBlock must be in (0, {BlockCount}) exclusive, got {splitBlock}.");
        if (BeforeBlockForward is not null)
            throw new InvalidOperationException(
                "DiT sharding and block streaming don't compose — BeforeBlockForward must be null on the sharded path.");

        int batch = (int)patchedLatent.Shape[0];
        int imgSeqLen = (int)patchedLatent.Shape[1];

        (Tensor currentImg, Tensor currentTxt, Tensor temb) = ForwardEmbedIn(
            backendA, patchedLatent, encoderHidden, encoderHidden2, timestep, guidanceScale);

        ForwardBlocksRange(backendA, ref currentImg, ref currentTxt, temb, postPatchH, postPatchW, 0, splitBlock);

        // Boundary A→B: both live streams plus the per-step conditioning every block reads. The streams MOVE
        // (A's copies are dead), but temb is only COPIED — the head on A still reads it after B's range. The
        // rope's host cos/sin tables need no copy — they are built host-side and WanRopeInterleaved stages them
        // on whichever backend runs.
        Tensor imgB = MoveAcross(backendB, backendA, currentImg);
        Tensor txtB = MoveAcross(backendB, backendA, currentTxt);
        Tensor tembB = CopyAcross(backendB, backendA, temb);
        currentImg = imgB;
        currentTxt = txtB;

        ForwardBlocksRange(backendB, ref currentImg, ref currentTxt, tembB, postPatchH, postPatchW, splitBlock, BlockCount);
        currentTxt.Dispose();
        tembB.Dispose();

        // Boundary B→A: the head (norm_out/proj_out) lives in the shared weights on A.
        Tensor imgBack = MoveAcross(backendA, backendB, currentImg);
        currentImg = imgBack;

        Tensor output = ForwardHeadOut(backendA, currentImg, temb, batch, imgSeqLen);
        temb.Dispose();
        return output;
    }

    /// <summary>Peer-copies <paramref name="source"/> onto <paramref name="dst"/>'s device and disposes the source.</summary>
    private static Tensor MoveAcross(IBackend dst, IBackend src, Tensor source)
    {
        Tensor moved = CopyAcross(dst, src, source);
        source.Dispose();
        return moved;
    }

    /// <summary>Peer-copies <paramref name="source"/> onto <paramref name="dst"/>'s device; the source stays live.</summary>
    private static Tensor CopyAcross(IBackend dst, IBackend src, Tensor source)
    {
        Tensor copied = new Tensor(source.Shape, source.DType);
        dst.CopyFromPeer(copied, source, src);
        return copied;
    }

    /// <summary>The pre-block section of <see cref="Forward"/>: x_embedder Linear, the timestep/guidance embedding, the token-refiner context embed, and the optional ByT5 concat into the txt stream.</summary>
    private (Tensor ImgTokens, Tensor TxtTokens, Tensor Temb) ForwardEmbedIn(IBackend backend, Tensor patchedLatent,
        Tensor encoderHidden, Tensor? encoderHidden2, float timestep, float guidanceScale)
    {
        int batch = (int)patchedLatent.Shape[0];
        int imgSeqLen = (int)patchedLatent.Shape[1];
        int txtSeqLen = (int)encoderHidden.Shape[1];
        int hidden = _config.HiddenSize;

        TensorShape imgTokShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor imgTokens = new Tensor(imgTokShape, DType.F32);
        backend.Linear(imgTokens, patchedLatent, _xEmbedWeight!, _xEmbedBias);
        HunyuanImageDebugDump.Dump("x_embedder", imgTokens);

        Tensor temb = ComputeTimestepGuidanceEmbedding(backend, timestep, guidanceScale, batch);
        HunyuanImageDebugDump.Dump("time_guidance_embed", temb);

        Tensor txtTokens = _contextEmbedder.Forward(backend, encoderHidden, timestep);
        HunyuanImageDebugDump.Dump("context_embedder", txtTokens);

        if (_contextEmbedder2 is not null && encoderHidden2 is not null)
        {
            Tensor txt2Tokens = _contextEmbedder2.Forward(backend, encoderHidden2);
            HunyuanImageDebugDump.Dump("context_embedder_2", txt2Tokens);

            int txt2SeqLen = (int)txt2Tokens.Shape[1];
            int combinedSeqLen = txtSeqLen + txt2SeqLen;
            TensorShape combinedShape = new TensorShape(batch, combinedSeqLen, hidden);
            Tensor combined = new Tensor(combinedShape, DType.F32);
            ConcatTokensSeqDim(combined, txt2Tokens, txtTokens, batch, txt2SeqLen, txtSeqLen, hidden);
            txt2Tokens.Dispose();
            txtTokens.Dispose();
            txtTokens = combined;
        }

        return (imgTokens, txtTokens, temb);
    }

    /// <summary>Runs flat blocks <c>[startBlock, endBlock)</c> (double-stream first, then single-stream — the same global indexing as <see cref="GetBlock"/> and <see cref="BeforeBlockForward"/>), advancing both streams in place.</summary>
    // HunyuanImage does NOT opt into the shared 16-bit hot path (unlike the other DiTs on HARTSY_DIT_F16 — see
    // DitDtype.CastStreamToAct). Its Qwen2.5-VL text conditioning is fed from an UN-normalized middle-layer
    // hidden state (hidden_states[-3]), which legitimately carries residual-stream magnitudes in the thousands
    // (measured ~3000 at context_embedder output for a real prompt, vs O(1-10) for the image stream). That
    // text stream keeps compounding through the 20 double-stream blocks' residual adds and — for a real
    // (non-degenerate) prompt specifically — overflows F16's 65504 ceiling by the last double block, producing
    // an Inf that poisons the very first single-stream block's joint self-attention (every query attends to
    // the same poisoned text key/value, so one Inf/NaN token contaminates the entire joint softmax) and
    // cascades to an all-NaN velocity → all-NaN latent → solid-black decoded image. A short negative/empty
    // prompt (1 text token, small magnitude) never reaches the overflow, which is why CFG's unconditional
    // branch always looked fine while the conditional branch went black. BF16 (F32's exponent range, smaller
    // mantissa) would dodge the overflow while keeping a 16-bit activation, but the shared LayerNormNoAffine /
    // AffineBroadcastLastDim CUDA kernels this arch's NormModulate depends on only accept F32 or F16 today —
    // adding BF16 support there is real follow-up work (multiple shared kernels, used by every other DiT), not
    // a small targeted fix. Running this transformer's block loop at F32 is: (a) always numerically safe (no
    // 16-bit ceiling to hit, matching the diffusers reference's own precision), and (b) fully local to this
    // file. DitRuntimeFlags.cs's own doc is explicit that opting a model into 16-bit activations requires a
    // per-arch safety audit — HunyuanImage's text-conditioning range fails that audit, so it stays F32.
    private void ForwardBlocksRange(IBackend backend, ref Tensor currentImg, ref Tensor currentTxt,
        Tensor temb, int postPatchH, int postPatchW, int startBlock, int endBlock)
    {
        for (int i = startBlock; i < endBlock; i++)
        {
            BeforeBlockForward?.Invoke(i);
            (Tensor newImg, Tensor newTxt) = i < _config.NumDoubleBlocks
                ? _doubleBlocks[i].Forward(
                    backend, currentImg, currentTxt, temb, _rope, postPatchH, postPatchW)
                : _singleBlocks[i - _config.NumDoubleBlocks].Forward(
                    backend, currentImg, currentTxt, temb, _rope, postPatchH, postPatchW);
            currentImg.Dispose();
            currentTxt.Dispose();
            currentImg = newImg;
            currentTxt = newTxt;
            if (i < _config.NumDoubleBlocks)
            {
                HunyuanImageDebugDump.Dump($"double_{i}_image", currentImg);
                HunyuanImageDebugDump.Dump($"double_{i}_text", currentTxt);
            }
            else
            {
                HunyuanImageDebugDump.Dump($"single_{i - _config.NumDoubleBlocks}_image", currentImg);
            }
        }
    }

    /// <summary>The post-block section of <see cref="Forward"/>: F32 restore, AdaLN-continuous final layer, and the debug dumps. Consumes <paramref name="currentImg"/>; the caller disposes <paramref name="temb"/>.</summary>
    private Tensor ForwardHeadOut(IBackend backend, Tensor currentImg, Tensor temb, int batch, int imgSeqLen)
    {
        // Back to F32 for the final AdaLN-continuous norm + proj_out (velocity precision matters across the Euler
        // steps, and ApplyFinalLayer's modulate reads the normed tensor on the host as float*). No-op when F16 is off.
        currentImg = Utilities.DtypeCastHelper.EnsureF32(backend, currentImg);

        Tensor output = ApplyFinalLayer(backend, currentImg, temb, batch, imgSeqLen);
        HunyuanImageDebugDump.Dump("proj_out", output);
        currentImg.Dispose();

        HunyuanImageDebugDump.DumpOutput(output);
        return output;
    }

    /// <summary>Computes the timestep + optional distilled-guidance conditioning vector. Diffusers' <c>HunyuanImageCombinedTimeGuidanceEmbedding</c>: sinusoidal(256, downscale_freq_shift=0) → timestep_embedder MLP (Linear→SiLU→Linear) producing temb_t. When <see cref="HunyuanImageConfig.GuidanceEmbed"/> is true, the same sinusoidal projection feeds a separate <c>guidance_embedder</c> MLP and the two are summed.</summary>
    private Tensor ComputeTimestepGuidanceEmbedding(IBackend backend, float timestep, float guidanceScale, int batch)
    {
        int hidden = _config.HiddenSize;

        TensorShape sinShape = new TensorShape(batch, 256);
        Tensor sinEmbed = new Tensor(sinShape, DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinEmbed, timestep, batch, embDim: 256);

        TensorShape hidShape = new TensorShape(batch, hidden);
        Tensor t1 = new Tensor(hidShape, DType.F32);
        backend.Linear(t1, sinEmbed, _timeProjLinear1Weight!, _timeProjLinear1Bias);

        Tensor t1Activated = new Tensor(hidShape, DType.F32);
        backend.Silu(t1Activated, t1);
        t1.Dispose();

        Tensor temb = new Tensor(hidShape, DType.F32);
        backend.Linear(temb, t1Activated, _timeProjLinear2Weight!, _timeProjLinear2Bias);
        t1Activated.Dispose();

        if (_config.GuidanceEmbed && _guidanceLinear1Weight is not null)
        {
            Tensor gSin = new Tensor(sinShape, DType.F32);
            DiTUtils.SinusoidalTimestepEmbedding(gSin, guidanceScale * 1000.0f, batch, embDim: 256);

            Tensor g1 = new Tensor(hidShape, DType.F32);
            backend.Linear(g1, gSin, _guidanceLinear1Weight!, _guidanceLinear1Bias);
            gSin.Dispose();

            Tensor g1Activated = new Tensor(hidShape, DType.F32);
            backend.Silu(g1Activated, g1);
            g1.Dispose();

            Tensor gEmb = new Tensor(hidShape, DType.F32);
            backend.Linear(gEmb, g1Activated, _guidanceLinear2Weight!, _guidanceLinear2Bias);
            g1Activated.Dispose();

            Tensor combined = new Tensor(hidShape, DType.F32);
            backend.Add(combined, temb, gEmb);
            temb.Dispose();
            gEmb.Dispose();
            temb = combined;
        }

        sinEmbed.Dispose();
        return temb;
    }

    /// <summary>Final AdaLN-continuous: <c>SiLU(temb) → Linear → [shift, scale]</c>, unparameterized LayerNorm + modulate, then <c>proj_out</c>. Diffusers' <c>AdaLayerNormContinuous</c> chunks <c>[scale, shift]</c> scale-first.</summary>
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
                    // Diffusers AdaLayerNormContinuous chunks [scale, shift] SCALE-FIRST (same as the
                    // Qwen-Image final layer; the conversion script reorders Tencent's [shift, scale]).
                    float scale = modPtr[modBase + d];
                    float shift = modPtr[modBase + dim + d];
                    outModPtr[vecOffset + d] = normPtr[vecOffset + d] * (1.0f + scale) + shift;
                }
            }
        }
        normed.Dispose();
        modParams.Dispose();
        HunyuanImageDebugDump.Dump("norm_out", modulated);

        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor projected = new Tensor(outShape, DType.F32);
        backend.Linear(projected, modulated, _projOutWeight!, _projOutBias);
        modulated.Dispose();

        return projected;
    }

    /// <summary>Pipeline-level debug hook: dumps the post-denoise, pre-VAE latent under <c>$HUNYUAN_IMAGE_DEBUG_DIR/final_latent.bin</c> when the env var is set.</summary>
    public static void DumpFinalLatent(Tensor latent) => HunyuanImageDebugDump.Dump("final_latent", latent);

    private static void ConcatTokensSeqDim(Tensor output, Tensor a, Tensor b,
        int batch, int seqA, int seqB, int dim)
    {
        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int totalSeq = seqA + seqB;

        for (int bi = 0; bi < batch; bi++)
        {
            long aBytes = (long)seqA * dim * sizeof(float);
            long bBytes = (long)seqB * dim * sizeof(float);
            Buffer.MemoryCopy(aPtr + (long)bi * seqA * dim,
                outPtr + (long)bi * totalSeq * dim, aBytes, aBytes);
            Buffer.MemoryCopy(bPtr + (long)bi * seqB * dim,
                outPtr + (long)bi * totalSeq * dim + seqA * dim, bBytes, bBytes);
        }
    }

    /// <summary>Releases all tensor references.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _xEmbedWeight = null; _xEmbedBias = null;
            _timeProjLinear1Weight = null; _timeProjLinear1Bias = null;
            _timeProjLinear2Weight = null; _timeProjLinear2Bias = null;
            _guidanceLinear1Weight = null; _guidanceLinear1Bias = null;
            _guidanceLinear2Weight = null; _guidanceLinear2Bias = null;
            _normOutLinearWeight = null; _normOutLinearBias = null;
            _projOutWeight = null; _projOutBias = null;
        }
        GC.SuppressFinalize(this);
    }
}

/// <summary>Hunyuan Image text token refiner (<c>HunyuanImageTokenRefiner</c>). Pools the encoder hidden states (mean over the sequence), combines with sinusoidal timestep via <c>CombinedTimestepTextProjEmbeddings</c>, projects each token to <see cref="HunyuanImageConfig.HiddenSize"/> through a Linear, then runs <see cref="NumLayers"/> self-attention refiner blocks. Each refiner block is LN→self-attn→AdaNorm-gated residual + LN→GELU FFN→AdaNorm-gated residual where the two gates come from a 2-param Linear-after-SiLU on the temb. Closes over the diffusers reference <c>HunyuanImageIndividualTokenRefinerBlock.forward</c>.</summary>
public sealed unsafe class HunyuanImageTokenRefiner
{
    private readonly int _inDim;
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _numLayers;
    private readonly int _mlpDim;
    private readonly float _qkNormEps;

    private Tensor? _timeTextEmbedTimestepLinear1Weight, _timeTextEmbedTimestepLinear1Bias;
    private Tensor? _timeTextEmbedTimestepLinear2Weight, _timeTextEmbedTimestepLinear2Bias;
    private Tensor? _timeTextEmbedTextLinear1Weight, _timeTextEmbedTextLinear1Bias;
    private Tensor? _timeTextEmbedTextLinear2Weight, _timeTextEmbedTextLinear2Bias;

    private Tensor? _projInWeight, _projInBias;

    private readonly RefinerBlock[] _blocks;

    public HunyuanImageTokenRefiner(int inDim, int hiddenSize, int numHeads, int headDim,
        int numLayers, float mlpRatio = 4.0f, float qkNormEps = 1e-6f)
    {
        _inDim = inDim;
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;
        _numLayers = numLayers;
        _mlpDim = (int)(hiddenSize * mlpRatio);
        _qkNormEps = qkNormEps;

        _blocks = new RefinerBlock[numLayers];
        for (int i = 0; i < numLayers; i++)
            _blocks[i] = new RefinerBlock(hiddenSize, numHeads, headDim, _mlpDim);
    }

    /// <summary>Number of refiner self-attention layers.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Loads weights under <paramref name="prefix"/>.* using diffusers naming.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _timeTextEmbedTimestepLinear1Weight = weights[$"{prefix}.time_text_embed.timestep_embedder.linear_1.weight"];
        _timeTextEmbedTimestepLinear1Bias = weights[$"{prefix}.time_text_embed.timestep_embedder.linear_1.bias"];
        _timeTextEmbedTimestepLinear2Weight = weights[$"{prefix}.time_text_embed.timestep_embedder.linear_2.weight"];
        _timeTextEmbedTimestepLinear2Bias = weights[$"{prefix}.time_text_embed.timestep_embedder.linear_2.bias"];
        _timeTextEmbedTextLinear1Weight = weights[$"{prefix}.time_text_embed.text_embedder.linear_1.weight"];
        _timeTextEmbedTextLinear1Bias = weights[$"{prefix}.time_text_embed.text_embedder.linear_1.bias"];
        _timeTextEmbedTextLinear2Weight = weights[$"{prefix}.time_text_embed.text_embedder.linear_2.weight"];
        _timeTextEmbedTextLinear2Bias = weights[$"{prefix}.time_text_embed.text_embedder.linear_2.bias"];

        _projInWeight = weights[$"{prefix}.proj_in.weight"];
        _projInBias = weights[$"{prefix}.proj_in.bias"];

        for (int i = 0; i < _numLayers; i++)
            _blocks[i].LoadWeights(weights, $"{prefix}.token_refiner.refiner_blocks.{i}");
    }

    /// <summary>Yields all refiner weights for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_timeTextEmbedTimestepLinear1Weight is not null) yield return _timeTextEmbedTimestepLinear1Weight;
        if (_timeTextEmbedTimestepLinear1Bias is not null) yield return _timeTextEmbedTimestepLinear1Bias;
        if (_timeTextEmbedTimestepLinear2Weight is not null) yield return _timeTextEmbedTimestepLinear2Weight;
        if (_timeTextEmbedTimestepLinear2Bias is not null) yield return _timeTextEmbedTimestepLinear2Bias;
        if (_timeTextEmbedTextLinear1Weight is not null) yield return _timeTextEmbedTextLinear1Weight;
        if (_timeTextEmbedTextLinear1Bias is not null) yield return _timeTextEmbedTextLinear1Bias;
        if (_timeTextEmbedTextLinear2Weight is not null) yield return _timeTextEmbedTextLinear2Weight;
        if (_timeTextEmbedTextLinear2Bias is not null) yield return _timeTextEmbedTextLinear2Bias;
        if (_projInWeight is not null) yield return _projInWeight;
        if (_projInBias is not null) yield return _projInBias;
        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;
    }

    /// <summary>Forward: pools the input over the sequence axis, projects to hidden, and runs N self-attn refiner blocks. Returns <c>[B, txtSeqLen, hidden_size]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor encoderHidden, float timestep)
    {
        int batch = (int)encoderHidden.Shape[0];
        int txtSeqLen = (int)encoderHidden.Shape[1];
        int inDim = (int)encoderHidden.Shape[2];

        TensorShape pooledShape = new TensorShape(batch, inDim);
        Tensor pooled = new Tensor(pooledShape, DType.F32);
        MeanPoolSeq(pooled, encoderHidden, batch, txtSeqLen, inDim);

        Tensor temb = ComputeRefinerTemb(backend, timestep, pooled, batch);
        pooled.Dispose();

        TensorShape projShape = new TensorShape(batch, txtSeqLen, _hiddenSize);
        Tensor projected = new Tensor(projShape, DType.F32);
        backend.Linear(projected, encoderHidden, _projInWeight!, _projInBias);

        Tensor current = projected;
        for (int i = 0; i < _numLayers; i++)
        {
            Tensor next = _blocks[i].Forward(backend, current, temb);
            current.Dispose();
            current = next;
        }

        temb.Dispose();
        return current;
    }

    private Tensor ComputeRefinerTemb(IBackend backend, float timestep, Tensor pooled, int batch)
    {
        int hidden = _hiddenSize;

        TensorShape sinShape = new TensorShape(batch, 256);
        Tensor sinEmbed = new Tensor(sinShape, DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinEmbed, timestep, batch, embDim: 256);

        TensorShape hidShape = new TensorShape(batch, hidden);
        Tensor t1 = new Tensor(hidShape, DType.F32);
        backend.Linear(t1, sinEmbed, _timeTextEmbedTimestepLinear1Weight!, _timeTextEmbedTimestepLinear1Bias);
        sinEmbed.Dispose();

        Tensor t1Activated = new Tensor(hidShape, DType.F32);
        backend.Silu(t1Activated, t1);
        t1.Dispose();

        Tensor tEmb = new Tensor(hidShape, DType.F32);
        backend.Linear(tEmb, t1Activated, _timeTextEmbedTimestepLinear2Weight!, _timeTextEmbedTimestepLinear2Bias);
        t1Activated.Dispose();

        Tensor p1 = new Tensor(hidShape, DType.F32);
        backend.Linear(p1, pooled, _timeTextEmbedTextLinear1Weight!, _timeTextEmbedTextLinear1Bias);
        Tensor p1Activated = new Tensor(hidShape, DType.F32);
        backend.Silu(p1Activated, p1);
        p1.Dispose();

        Tensor pEmb = new Tensor(hidShape, DType.F32);
        backend.Linear(pEmb, p1Activated, _timeTextEmbedTextLinear2Weight!, _timeTextEmbedTextLinear2Bias);
        p1Activated.Dispose();

        Tensor combined = new Tensor(hidShape, DType.F32);
        backend.Add(combined, tEmb, pEmb);
        tEmb.Dispose();
        pEmb.Dispose();
        return combined;
    }

    private static void MeanPoolSeq(Tensor output, Tensor input, int batch, int seqLen, int dim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float invLen = 1.0f / seqLen;

        for (int b = 0; b < batch; b++)
        {
            for (int d = 0; d < dim; d++)
            {
                float sum = 0f;
                for (int s = 0; s < seqLen; s++)
                    sum += inPtr[(b * seqLen + s) * dim + d];
                outPtr[b * dim + d] = sum * invLen;
            }
        }
    }

    /// <summary>One refiner block: <c>norm1 → self-attn (with norm_q/norm_k QK-RMSNorm) → gate_msa-modulated residual → norm2 → linear-silu FFN → gate_mlp-modulated residual</c>. Both gates come from <c>norm_out.linear(silu(temb))</c> chunked into 2 hidden-sized halves. Diffusers <c>HunyuanImageIndividualTokenRefinerBlock</c> with FeedForward <c>activation_fn="linear-silu"</c> = <c>SiLU(W1 x) * (W3 x) → W2</c>.</summary>
    public sealed class RefinerBlock
    {
        private readonly int _hiddenSize;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly int _mlpDim;

        private Tensor? _norm1Weight, _norm1Bias;
        private Tensor? _norm2Weight, _norm2Bias;
        private Tensor? _attnQWeight, _attnQBias;
        private Tensor? _attnKWeight, _attnKBias;
        private Tensor? _attnVWeight, _attnVBias;
        private Tensor? _attnOutWeight, _attnOutBias;
        private Tensor? _ffN0ProjWeight, _ffN0ProjBias;
        private Tensor? _ffN0LinearWeight, _ffN0LinearBias;
        private Tensor? _ffN2Weight, _ffN2Bias;
        private Tensor? _normOutLinearWeight, _normOutLinearBias;

        public RefinerBlock(int hiddenSize, int numHeads, int headDim, int mlpDim)
        {
            _hiddenSize = hiddenSize;
            _numHeads = numHeads;
            _headDim = headDim;
            _mlpDim = mlpDim;
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
        {
            _norm1Weight = weights[$"{prefix}.norm1.weight"];
            _norm1Bias = weights[$"{prefix}.norm1.bias"];
            _norm2Weight = weights[$"{prefix}.norm2.weight"];
            _norm2Bias = weights[$"{prefix}.norm2.bias"];

            _attnQWeight = weights[$"{prefix}.attn.to_q.weight"];
            _attnKWeight = weights[$"{prefix}.attn.to_k.weight"];
            _attnVWeight = weights[$"{prefix}.attn.to_v.weight"];
            _attnOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];
            _attnOutBias = weights[$"{prefix}.attn.to_out.0.bias"];
            weights.TryGetValue($"{prefix}.attn.to_q.bias", out _attnQBias);
            weights.TryGetValue($"{prefix}.attn.to_k.bias", out _attnKBias);
            weights.TryGetValue($"{prefix}.attn.to_v.bias", out _attnVBias);

            _ffN0ProjWeight = weights[$"{prefix}.ff.net.0.proj.weight"];
            _ffN0ProjBias = weights[$"{prefix}.ff.net.0.proj.bias"];
            weights.TryGetValue($"{prefix}.ff.net.0.linear.weight", out _ffN0LinearWeight);
            weights.TryGetValue($"{prefix}.ff.net.0.linear.bias", out _ffN0LinearBias);
            _ffN2Weight = weights[$"{prefix}.ff.net.2.weight"];
            _ffN2Bias = weights[$"{prefix}.ff.net.2.bias"];

            _normOutLinearWeight = weights[$"{prefix}.norm_out.linear.weight"];
            _normOutLinearBias = weights[$"{prefix}.norm_out.linear.bias"];
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            if (_norm1Weight is not null) yield return _norm1Weight;
            if (_norm1Bias is not null) yield return _norm1Bias;
            if (_norm2Weight is not null) yield return _norm2Weight;
            if (_norm2Bias is not null) yield return _norm2Bias;
            if (_attnQWeight is not null) yield return _attnQWeight;
            if (_attnQBias is not null) yield return _attnQBias;
            if (_attnKWeight is not null) yield return _attnKWeight;
            if (_attnKBias is not null) yield return _attnKBias;
            if (_attnVWeight is not null) yield return _attnVWeight;
            if (_attnVBias is not null) yield return _attnVBias;
            if (_attnOutWeight is not null) yield return _attnOutWeight;
            if (_attnOutBias is not null) yield return _attnOutBias;
            if (_ffN0ProjWeight is not null) yield return _ffN0ProjWeight;
            if (_ffN0ProjBias is not null) yield return _ffN0ProjBias;
            if (_ffN0LinearWeight is not null) yield return _ffN0LinearWeight;
            if (_ffN0LinearBias is not null) yield return _ffN0LinearBias;
            if (_ffN2Weight is not null) yield return _ffN2Weight;
            if (_ffN2Bias is not null) yield return _ffN2Bias;
            if (_normOutLinearWeight is not null) yield return _normOutLinearWeight;
            if (_normOutLinearBias is not null) yield return _normOutLinearBias;
        }

        public Tensor Forward(IBackend backend, Tensor x, Tensor temb)
        {
            int batch = (int)x.Shape[0];
            int seqLen = (int)x.Shape[1];
            TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);

            Tensor normed1 = new Tensor(shape, DType.F32);
            backend.LayerNorm(normed1, x, _norm1Weight!, _norm1Bias!, 1e-6f);

            Tensor q = new Tensor(shape, DType.F32);
            backend.Linear(q, normed1, _attnQWeight!, _attnQBias);
            Tensor k = new Tensor(shape, DType.F32);
            backend.Linear(k, normed1, _attnKWeight!, _attnKBias);
            Tensor v = new Tensor(shape, DType.F32);
            backend.Linear(v, normed1, _attnVWeight!, _attnVBias);
            normed1.Dispose();

            TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
            Tensor qMh = new Tensor(mhShape, DType.F32);
            Tensor kMh = new Tensor(mhShape, DType.F32);
            Tensor vMh = new Tensor(mhShape, DType.F32);
            DiTUtils.ReshapeToMultiHead(qMh, q, batch, seqLen, _numHeads, _headDim);
            DiTUtils.ReshapeToMultiHead(kMh, k, batch, seqLen, _numHeads, _headDim);
            DiTUtils.ReshapeToMultiHead(vMh, v, batch, seqLen, _numHeads, _headDim);
            q.Dispose();
            k.Dispose();
            v.Dispose();

            float scale = 1.0f / MathF.Sqrt(_headDim);
            Tensor attnOut = new Tensor(mhShape, DType.F32);
            backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, null, scale);
            qMh.Dispose();
            kMh.Dispose();
            vMh.Dispose();

            Tensor attnFlat = new Tensor(shape, DType.F32);
            DiTUtils.ReshapeFromMultiHead(attnFlat, attnOut, batch, seqLen, _numHeads, _headDim);
            attnOut.Dispose();

            Tensor attnProj = new Tensor(shape, DType.F32);
            backend.Linear(attnProj, attnFlat, _attnOutWeight!, _attnOutBias);
            attnFlat.Dispose();

            (Tensor gateMsa, Tensor gateMlp) = ComputeGates(backend, temb, batch);

            Tensor afterAttn = AdaLNModulation.ApplyGatedResidual(x, attnProj, gateMsa, batch, seqLen, _hiddenSize);
            attnProj.Dispose();

            Tensor normed2 = new Tensor(shape, DType.F32);
            backend.LayerNorm(normed2, afterAttn, _norm2Weight!, _norm2Bias!, 1e-6f);

            Tensor mlpOut = ApplyLinearSiluFfn(backend, normed2, batch, seqLen);
            normed2.Dispose();

            Tensor result = AdaLNModulation.ApplyGatedResidual(afterAttn, mlpOut, gateMlp, batch, seqLen, _hiddenSize);
            afterAttn.Dispose();
            mlpOut.Dispose();
            gateMsa.Dispose();
            gateMlp.Dispose();

            return result;
        }

        /// <summary>Computes the two AdaNorm gates from the refiner block's <c>norm_out.linear(silu(temb))</c>. Output of the linear is <c>[B, 2*hidden]</c>; chunk in order <c>(gate_msa, gate_mlp)</c>.</summary>
        private unsafe (Tensor msa, Tensor mlp) ComputeGates(IBackend backend, Tensor temb, int batch)
        {
            int hidden = _hiddenSize;
            TensorShape inShape = new TensorShape(batch, hidden);
            Tensor activated = new Tensor(inShape, DType.F32);
            backend.Silu(activated, temb);

            TensorShape outShape = new TensorShape(batch, 2 * hidden);
            Tensor proj = new Tensor(outShape, DType.F32);
            backend.Linear(proj, activated, _normOutLinearWeight!, _normOutLinearBias);
            activated.Dispose();

            Tensor msa = new Tensor(inShape, DType.F32);
            Tensor mlp = new Tensor(inShape, DType.F32);
            float* projPtr = (float*)proj.DataPointer;
            float* msaPtr = (float*)msa.DataPointer;
            float* mlpPtr = (float*)mlp.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                int srcBase = b * 2 * hidden;
                int dstBase = b * hidden;
                for (int d = 0; d < hidden; d++)
                {
                    msaPtr[dstBase + d] = projPtr[srcBase + d];
                    mlpPtr[dstBase + d] = projPtr[srcBase + hidden + d];
                }
            }
            proj.Dispose();
            return (msa, mlp);
        }

        /// <summary>Linear-SiLU FFN as used by diffusers' <c>FeedForward(activation_fn="linear-silu")</c>: <c>output = W2(SiLU(W1(x)) * Wlin(x))</c> when a separate <c>linear</c> branch is present, otherwise <c>output = W2(SiLU(W1(x)))</c>. We detect the gated form by the presence of <c>ff.net.0.linear.weight</c>.</summary>
        private Tensor ApplyLinearSiluFfn(IBackend backend, Tensor input, int batch, int seqLen)
        {
            TensorShape ffShape = new TensorShape(batch, seqLen, _mlpDim);
            Tensor proj = new Tensor(ffShape, DType.F32);
            backend.Linear(proj, input, _ffN0ProjWeight!, _ffN0ProjBias);

            Tensor activated = new Tensor(ffShape, DType.F32);
            backend.Silu(activated, proj);
            proj.Dispose();

            if (_ffN0LinearWeight is not null)
            {
                Tensor linear = new Tensor(ffShape, DType.F32);
                backend.Linear(linear, input, _ffN0LinearWeight!, _ffN0LinearBias);
                Tensor gated = new Tensor(ffShape, DType.F32);
                backend.Mul(gated, activated, linear);
                activated.Dispose();
                linear.Dispose();
                activated = gated;
            }

            TensorShape outShape = new TensorShape(batch, seqLen, _hiddenSize);
            Tensor output = new Tensor(outShape, DType.F32);
            backend.Linear(output, activated, _ffN2Weight!, _ffN2Bias);
            activated.Dispose();
            return output;
        }
    }
}

/// <summary>Hunyuan Image ByT5 secondary text projection (<c>HunyuanImageByT5TextProjection</c>). LayerNorm → Linear(in, intermediate) → GELU → Linear(intermediate, intermediate) → GELU → Linear(intermediate, out). Used to project ByT5 glyph encoder hidden states into the transformer hidden dim before they are concatenated with the MLLM tokens.</summary>
public sealed class HunyuanImageByT5Projection(int inFeatures, int intermediateDim, int outFeatures)
{
    private readonly int _inFeatures = inFeatures;
    private readonly int _intermediateDim = intermediateDim;
    private readonly int _outFeatures = outFeatures;

    private Tensor? _normWeight, _normBias;
    private Tensor? _linear1Weight, _linear1Bias;
    private Tensor? _linear2Weight, _linear2Bias;
    private Tensor? _linear3Weight, _linear3Bias;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _normWeight = weights[$"{prefix}.norm.weight"];
        _normBias = weights[$"{prefix}.norm.bias"];
        _linear1Weight = weights[$"{prefix}.linear_1.weight"];
        _linear1Bias = weights[$"{prefix}.linear_1.bias"];
        _linear2Weight = weights[$"{prefix}.linear_2.weight"];
        _linear2Bias = weights[$"{prefix}.linear_2.bias"];
        _linear3Weight = weights[$"{prefix}.linear_3.weight"];
        _linear3Bias = weights[$"{prefix}.linear_3.bias"];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_normWeight is not null) yield return _normWeight;
        if (_normBias is not null) yield return _normBias;
        if (_linear1Weight is not null) yield return _linear1Weight;
        if (_linear1Bias is not null) yield return _linear1Bias;
        if (_linear2Weight is not null) yield return _linear2Weight;
        if (_linear2Bias is not null) yield return _linear2Bias;
        if (_linear3Weight is not null) yield return _linear3Weight;
        if (_linear3Bias is not null) yield return _linear3Bias;
    }

    public Tensor Forward(IBackend backend, Tensor encoderHidden)
    {
        int batch = (int)encoderHidden.Shape[0];
        int seqLen = (int)encoderHidden.Shape[1];

        TensorShape inShape = new TensorShape(batch, seqLen, _inFeatures);
        Tensor normed = new Tensor(inShape, DType.F32);
        backend.LayerNorm(normed, encoderHidden, _normWeight!, _normBias!, 1e-6f);

        TensorShape midShape = new TensorShape(batch, seqLen, _intermediateDim);
        Tensor h1 = new Tensor(midShape, DType.F32);
        backend.Linear(h1, normed, _linear1Weight!, _linear1Bias);
        normed.Dispose();

        Tensor h1Act = new Tensor(midShape, DType.F32);
        backend.Gelu(h1Act, h1);
        h1.Dispose();

        Tensor h2 = new Tensor(midShape, DType.F32);
        backend.Linear(h2, h1Act, _linear2Weight!, _linear2Bias);
        h1Act.Dispose();

        Tensor h2Act = new Tensor(midShape, DType.F32);
        backend.Gelu(h2Act, h2);
        h2.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, _outFeatures);
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Linear(output, h2Act, _linear3Weight!, _linear3Bias);
        h2Act.Dispose();
        return output;
    }
}
