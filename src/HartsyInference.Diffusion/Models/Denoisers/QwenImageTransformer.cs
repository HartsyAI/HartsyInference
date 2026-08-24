using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
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

        _rope = new QwenImageRope(theta: config.RopeTheta, ropeText: config.RopeText);
    }

    /// <summary>Loads all transformer weights from named tensors using diffusers naming.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        // F32 activations only: this model's massive-activation channels push the residual stream to ±10M by
        // mid-depth (measured block-input absmax) — 150× past F16 range, which is why reference stacks run it
        // in BF16. A damped F16 loop was tried and removed: the second denoise step NaN'd deterministically.
        _imgInWeight = weights["img_in.weight"];
        _imgInBias = weights["img_in.bias"];

        _txtNormWeight = TensorCasts.EnsureF32(weights["txt_norm.weight"]);

        _txtInWeight = weights["txt_in.weight"];
        _txtInBias = weights["txt_in.bias"];

        _timestepLinear1Weight = weights["time_text_embed.timestep_embedder.linear_1.weight"];
        _timestepLinear1Bias = weights["time_text_embed.timestep_embedder.linear_1.bias"];
        _timestepLinear2Weight = weights["time_text_embed.timestep_embedder.linear_2.weight"];
        _timestepLinear2Bias = weights["time_text_embed.timestep_embedder.linear_2.bias"];

        for (int i = 0; i < _config.Depth; i++)
            _blocks[i].LoadWeights(weights, $"transformer_blocks.{i}");

        _normOutLinearWeight = weights["norm_out.linear.weight"];
        _normOutLinearBias = weights["norm_out.linear.bias"];
        _projOutWeight = weights["proj_out.weight"];
        _projOutBias = weights["proj_out.bias"];
    }

    /// <summary>The number of streamable transformer blocks.</summary>
    public int BlockCount => _blocks.Length;

    /// <summary>The streamable block at <paramref name="idx"/>.</summary>
    /// <remarks>Returns the live block instance so <see cref="IStreamingBlock.EnumerateWeights"/> hands back the same
    /// tensor references every call — <c>BlockStreamingController</c> tracks residency by reference.</remarks>
    public IStreamingBlock GetBlock(int idx) => _blocks[idx];

    /// <summary>Hook invoked with each block index immediately before that block's forward pass; null = no streaming.</summary>
    /// <remarks>Pipelines plug a <c>BlockStreamingController</c> here so resident VRAM peaks at roughly
    /// (activations + the prefetch window) instead of the whole 20B DiT.
    /// <para><b>Any captured-graph fast path added later MUST be disabled while this is non-null</b> — a graph bakes
    /// weight device pointers and streaming re-points them every forward, so replay would read freed memory
    /// (context-poisoning CUDA 700). See <c>HunyuanVideoDit</c> / <c>LtxVideo2Transformer</c>. No graph path exists
    /// here today.</para></remarks>
    public Action<int>? BeforeBlockForward { get; set; }

    /// <summary>The non-block weights: the image/text input projections, the timestep MLP, and the output head.</summary>
    /// <remarks>Touched on every forward regardless of which block runs, so they stay eagerly resident even when
    /// streaming. Note these bracket the blocks in <see cref="EnumerateWeights"/> — the input projections come before
    /// and the output head after — so this is a genuine partition, not a prefix.</remarks>
    public IEnumerable<Tensor> EnumerateSharedWeights()
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
        if (_normOutLinearWeight is not null) yield return _normOutLinearWeight;
        if (_normOutLinearBias is not null) yield return _normOutLinearBias;
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
    }

    /// <summary>Weights of blocks <c>[startBlock, endBlock)</c> only — the asymmetric-preload primitive for DiT
    /// sharding: backend A preloads <see cref="EnumerateSharedWeights"/> + its range, backend B ONLY its range.
    /// Never preload <see cref="EnumerateWeights"/> on both backends — that replicates instead of pooling.</summary>
    public IEnumerable<Tensor> EnumerateBlockRangeWeights(int startBlock, int endBlock)
    {
        for (int i = startBlock; i < endBlock; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;
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
    /// <param name="cp">Context-parallel rank context: the img token sequence is sliced to this rank's row range
    /// (the txt stream stays fully replicated — both ranks compute it deterministically), per-block joint-attention
    /// K/V is traded through <see cref="Utilities.CpKvExchange"/>, and the return value is the LOCAL packed velocity
    /// rows <c>[1, Sr, patch²·out]</c> — the caller gathers ranks in global token order. Excludes the step cache,
    /// block streaming, and edit-reference conditioning (the pipeline gates those off). Null keeps the
    /// single-backend path byte-identical.</param>
    public Tensor Forward(IBackend backend, Tensor packedLatent, Tensor encoderHidden, float timestep,
        int hPacked, int wPacked, (int H, int W)[]? refGrids = null, bool refTimestepZero = false,
        DeviceFeatureCache? stepCache = null, CpForwardContext? cp = null,
        Adapters.QwenImageControlNetResiduals? controlNetResiduals = null)
    {
        if (controlNetResiduals is not null)
        {
            // The pipeline gates these off when ControlNet is active (Flux parity): the step cache's
            // block-0-indicator/residual reconstruction skips the per-block adds, CP splits the img rows the
            // residuals cover, and edit-ref tokens change the img sequence the residuals were computed for.
            if (stepCache is not null)
                throw new InvalidOperationException("ControlNet residuals and the step cache don't compose — the pipeline must gate the cache off.");
            if (cp is not null)
                throw new InvalidOperationException("ControlNet residuals and context parallelism don't compose — the pipeline must gate CP off.");
            if (refGrids is { Length: > 0 })
                throw new InvalidOperationException("ControlNet residuals and edit-reference tokens don't compose — the pipeline must reject the combination.");
        }
        int batch = (int)packedLatent.Shape[0];
        int imgSeqLen = (int)packedLatent.Shape[1];
        int txtSeqLen = (int)encoderHidden.Shape[1];
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
        if (cp is not null)
        {
            if (stepCache is not null)
                throw new InvalidOperationException("Context parallelism excludes the step cache — the pipeline must gate it off.");
            if (BeforeBlockForward is not null)
                throw new InvalidOperationException("Context parallelism and block streaming don't compose — BeforeBlockForward must be null.");
            if (refSeqLen > 0 || batch != 1)
                throw new InvalidOperationException("Context parallelism requires batch 1 and no edit-reference tokens — the pipeline must gate these off.");
            if (cp.Plan.TotalTokens != imgSeqLen || cp.Plan.TokensPerFrame != wPacked)
                throw new ArgumentException(
                    $"CP plan ({cp.Plan.Frames} rows × {cp.Plan.TokensPerFrame}) does not match the packed grid ({hPacked}×{wPacked}).", nameof(cp));
        }

        // Context parallel: this rank embeds and denoises only its row-aligned img token range (img_in is
        // per-token, so slicing the packed input first is exact); the txt stream runs full-length on every rank.
        Tensor embedSource = packedLatent;
        Tensor? localPacked = null;
        if (cp is not null)
        {
            CpRankRange range = cp.Plan.Ranks[cp.Rank];
            localPacked = new Tensor(new TensorShape(batch, range.TokenCount, packedLatent.Shape[2]), DType.F32);
            backend.SliceRows(localPacked, packedLatent, range.TokenStart);
            embedSource = localPacked;
            imgSeqLen = range.TokenCount;
        }

        (Tensor imgTokens, Tensor txtTokens) = ForwardEmbedIn(backend, embedSource, encoderHidden, batch, imgSeqLen, txtSeqLen);
        localPacked?.Dispose();

        Tensor temb = ComputeTimestepEmbedding(backend, timestep, batch);
        QwenImageDebugDump.Dump("time_text_embed", temb);

        // 2511 timestep-zero ref method: a second temb at t=0 drives the ref-row modulation in every block
        // (identical to upstream's batch-2 `cat([timestep, timestep*0])` — separate embedding call, same math).
        Tensor? tembZero = refSeqLen > 0 && refTimestepZero ? ComputeTimestepEmbedding(backend, 0.0f, batch) : null;

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
            BeforeBlockForward?.Invoke(0);
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

        // A step-cache HIT set startBlock = Depth, so this range is empty and its streaming hook never fires. The
        // controller simply keeps whatever it had prefetched (bounded by the prefetch window) and the next miss
        // resumes from there — residency is per-block state, not a position in a sequence.
        ForwardBlocksRange(backend, ref currentImg, ref currentTxt, temb, tembZero, hPacked, wPacked, txtPositionStart,
            refGrids, mainSeqLen, startBlock, _config.Depth, cacheAnchor, cp, controlNetResiduals?.BlockResiduals);

        if (cacheAnchor is not null)
        {
            stepCache!.StoreResidual(backend, cacheAnchor, currentImg);
            cacheAnchor.Dispose();
        }

        currentTxt.Dispose();

        Tensor output = ForwardHeadOut(backend, currentImg, temb, batch, imgSeqLen, refSeqLen, mainSeqLen);
        temb.Dispose();
        tembZero?.Dispose();
        return output;
    }

    /// <summary>DiT-sharded forward: blocks <c>[0, splitBlock)</c> on <paramref name="backendA"/> (which also owns
    /// the shared embed/head weights), <c>[splitBlock, Depth)</c> on <paramref name="backendB"/>, with the img+txt
    /// streams, temb, and (edit) tembZero handed across via <see cref="IBackend.CopyFromPeer"/> and the img stream
    /// handed back for the head. VRAM pooling, not latency — the two backends run sequentially. Exclusions mirror
    /// Krea2: no step-cache (its block-0-indicator shape doesn't compose with a fixed range boundary), no
    /// step-graph (none exists here), no block streaming (<see cref="BeforeBlockForward"/> must be null — the
    /// sharding preload owns block residency).
    /// Callers preload <see cref="EnumerateSharedWeights"/> + <see cref="EnumerateBlockRangeWeights"/>(0, split)
    /// on A and <see cref="EnumerateBlockRangeWeights"/>(split, Depth) on B.</summary>
    public Tensor ForwardSharded(IBackend backendA, IBackend backendB, Tensor packedLatent, Tensor encoderHidden,
        float timestep, int hPacked, int wPacked, int splitBlock,
        (int H, int W)[]? refGrids = null, bool refTimestepZero = false)
    {
        if (splitBlock <= 0 || splitBlock >= _blocks.Length)
            throw new ArgumentOutOfRangeException(nameof(splitBlock),
                $"splitBlock must be in (0, {_blocks.Length}) exclusive, got {splitBlock}.");

        return ForwardSharded(
            [new DitShardStage(backendA, 0, splitBlock), new DitShardStage(backendB, splitBlock, _blocks.Length)],
            packedLatent, encoderHidden, timestep, hPacked, wPacked, refGrids, refTimestepZero);
    }

    /// <summary>N-way generalization of <see cref="ForwardSharded(IBackend,IBackend,Tensor,Tensor,float,int,int,int,ValueTuple{int,int}[],bool)"/>:
    /// an ordered list of <see cref="DitShardStage"/> block ranges, each on its own backend. Stage 0's backend owns
    /// the shared embed/head weights (mirrors the 2-way overload's backendA); the LAST stage hands the img stream
    /// back to stage 0 for the head. Every interior boundary peer-copies img+txt+temb(+tembZero) forward via
    /// <see cref="IBackend.CopyFromPeer"/> — temb/tembZero are copied fresh from stage 0's master copy at each
    /// boundary (not chained stage-to-stage) since they never change across blocks. A stage sharing its backend
    /// INSTANCE with stage 0 (e.g. two <c>ShardDevices</c> entries that canonically resolve to the same pooled
    /// backend) skips the temb copy and reuses the master tensor directly — see <c>InferenceEngine.EnsureDitShardBackends</c>
    /// for when that happens. The 2-way overload is exactly the N=2 case of this loop, so same-device bit-parity
    /// (<c>QwenImageDitShardingTests</c>) holds for both.</summary>
    public Tensor ForwardSharded(IReadOnlyList<DitShardStage> stages, Tensor packedLatent, Tensor encoderHidden,
        float timestep, int hPacked, int wPacked, (int H, int W)[]? refGrids = null, bool refTimestepZero = false)
    {
        ArgumentNullException.ThrowIfNull(stages);
        if (stages.Count < 2)
            throw new ArgumentException($"DiT sharding needs at least 2 stages, got {stages.Count}.", nameof(stages));
        int expected = 0;
        foreach (DitShardStage stage in stages)
        {
            ArgumentNullException.ThrowIfNull(stage.Backend);
            if (stage.StartBlock != expected || stage.EndBlock <= stage.StartBlock)
                throw new ArgumentException(
                    $"Stage ranges must be contiguous ascending starting at 0; got [{stage.StartBlock}, {stage.EndBlock}) after {expected}.",
                    nameof(stages));
            expected = stage.EndBlock;
        }
        if (expected != _blocks.Length)
            throw new ArgumentException($"Stage ranges must tile all {_blocks.Length} blocks; got {expected}.", nameof(stages));
        if (BeforeBlockForward is not null)
            throw new InvalidOperationException(
                "DiT sharding and block streaming don't compose — BeforeBlockForward must be null on the sharded path.");

        IBackend firstBackend = stages[0].Backend;
        int batch = (int)packedLatent.Shape[0];
        int imgSeqLen = (int)packedLatent.Shape[1];
        int txtSeqLen = (int)encoderHidden.Shape[1];
        refGrids ??= [];
        int refSeqLen = 0;
        foreach ((int rh, int rw) in refGrids) refSeqLen += rh * rw;
        int mainSeqLen = imgSeqLen - refSeqLen;
        if (refSeqLen > 0 && mainSeqLen != hPacked * wPacked)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                $"packedLatent seq {imgSeqLen} must equal main ({hPacked}x{wPacked}) + {refSeqLen} ref tokens.");
        if (refSeqLen > 0 && batch != 1)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                "Reference-latent (edit) tokens require batch size 1; run CFG as two batch-1 passes.");

        (Tensor currentImg, Tensor currentTxt) = ForwardEmbedIn(firstBackend, packedLatent, encoderHidden, batch, imgSeqLen, txtSeqLen);
        Tensor temb = ComputeTimestepEmbedding(firstBackend, timestep, batch);
        Tensor? tembZero = refSeqLen > 0 && refTimestepZero ? ComputeTimestepEmbedding(firstBackend, 0.0f, batch)
            : null;
        int txtPositionStart = QwenImageRope.ComputeTextPositionStart(hPacked, wPacked);

        for (int s = 0; s < stages.Count; s++)
        {
            DitShardStage stage = stages[s];
            bool sameAsFirst = s == 0 || ReferenceEquals(stage.Backend, firstBackend);
            Tensor stageTemb = sameAsFirst ? temb : CopyAcross(stage.Backend, firstBackend, temb);
            Tensor? stageTembZero = tembZero is null ? null : sameAsFirst ? tembZero : CopyAcross(stage.Backend, firstBackend, tembZero);

            ForwardBlocksRange(stage.Backend, ref currentImg, ref currentTxt, stageTemb, stageTembZero,
                hPacked, wPacked, txtPositionStart, refGrids, mainSeqLen, stage.StartBlock, stage.EndBlock);

            if (!sameAsFirst)
            {
                stageTemb.Dispose();
                stageTembZero?.Dispose();
            }

            bool isLast = s == stages.Count - 1;
            if (!isLast)
            {
                IBackend next = stages[s + 1].Backend;
                if (!ReferenceEquals(next, stage.Backend))
                {
                    currentImg = MoveAcross(next, stage.Backend, currentImg);
                    currentTxt = MoveAcross(next, stage.Backend, currentTxt);
                }
            }
        }
        currentTxt.Dispose();

        // The head (norm_out/proj_out) lives in the shared weights on stage 0's backend.
        IBackend lastBackend = stages[^1].Backend;
        if (!ReferenceEquals(lastBackend, firstBackend))
        {
            currentImg = MoveAcross(firstBackend, lastBackend, currentImg);
        }

        Tensor output = ForwardHeadOut(firstBackend, currentImg, temb, batch, imgSeqLen, refSeqLen, mainSeqLen);
        temb.Dispose();
        tembZero?.Dispose();
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

    /// <summary>The pre-block section of <see cref="Forward"/>: img_in Linear, txt RMSNorm + txt_in Linear.</summary>
    private (Tensor ImgTokens, Tensor TxtTokens) ForwardEmbedIn(IBackend backend, Tensor packedLatent,
        Tensor encoderHidden, int batch, int imgSeqLen, int txtSeqLen)
    {
        int hidden = _config.HiddenSize;
        Tensor imgTokens = new Tensor(new TensorShape(batch, imgSeqLen, hidden), DType.F32);
        backend.Linear(imgTokens, packedLatent, _imgInWeight!, _imgInBias);
        QwenImageDebugDump.Dump("img_in", imgTokens);

        Tensor txtNormed = new Tensor(encoderHidden.Shape, DType.F32);
        backend.RmsNorm(txtNormed, encoderHidden, _txtNormWeight!, 1e-6f);

        Tensor txtTokens = new Tensor(new TensorShape(batch, txtSeqLen, hidden), DType.F32);
        backend.Linear(txtTokens, txtNormed, _txtInWeight!, _txtInBias);
        txtNormed.Dispose();
        QwenImageDebugDump.Dump("txt_in", txtTokens);
        return (imgTokens, txtTokens);
    }

    /// <summary>Runs blocks <c>[startBlock, endBlock)</c>, advancing both streams in place. <paramref name="cacheAnchor"/>
    /// (step-cache anchor, the block-0 output) is exempt from the dispose-and-swap so <see cref="Forward"/> can store
    /// its residual after the loop; the sharded path passes null.</summary>
    private void ForwardBlocksRange(IBackend backend, ref Tensor currentImg, ref Tensor currentTxt,
        Tensor temb, Tensor? tembZero, int hPacked, int wPacked, int txtPositionStart,
        (int H, int W)[] refGrids, int mainSeqLen, int startBlock, int endBlock, Tensor? cacheAnchor = null,
        CpForwardContext? cp = null, Tensor[]? cnResiduals = null)
    {
        int cnInterval = cnResiduals is { Length: > 0 } ? (int)Math.Ceiling((double)_config.Depth / cnResiduals.Length)
            : 0;
        for (int i = startBlock; i < endBlock; i++)
        {
            BeforeBlockForward?.Invoke(i);
            (Tensor newImg, Tensor newTxt) = _blocks[i].Forward(
                backend, currentImg, currentTxt, temb, _rope,
                hPacked, wPacked, txtPositionStart, refGrids, tembZero, mainSeqLen, cp);

            if (currentImg != cacheAnchor) currentImg.Dispose();
            currentTxt.Dispose();

            currentImg = newImg;
            currentTxt = newTxt;

            // ControlNet residual (diffusers interval mapping: base block i ← residual i / ceil(depth/count)).
            if (cnInterval > 0)
            {
                Tensor res = cnResiduals![Math.Min(i / cnInterval, cnResiduals.Length - 1)];
                Tensor injected = new Tensor(currentImg.Shape, currentImg.DType);
                backend.Add(injected, currentImg, res);
                currentImg.Dispose();
                currentImg = injected;
            }

            QwenImageDebugDump.Dump($"block_{i}_image", currentImg);
            QwenImageDebugDump.Dump($"block_{i}_text", currentTxt);
        }
    }

    /// <summary>The post-block section of <see cref="Forward"/>: AdaLN-continuous final layer and the edit-mode
    /// reference-row drop. Consumes <paramref name="currentImg"/>.</summary>
    private Tensor ForwardHeadOut(IBackend backend, Tensor currentImg, Tensor temb, int batch, int imgSeqLen,
        int refSeqLen, int mainSeqLen)
    {
        Tensor output = ApplyFinalLayer(backend, currentImg, temb, batch, imgSeqLen);
        QwenImageDebugDump.Dump("proj_out", output);
        currentImg.Dispose();

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
