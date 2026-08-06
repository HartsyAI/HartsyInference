using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>SD3 / SD3.5 Multi-Modal Diffusion Transformer (MMDiT). Jointly processes image patches and text tokens through symmetric dual-stream attention. SD3.5 MMDiT-X variant marks the early layers for parallel image-only `attn2` paths via `Sd3Config.DualAttentionLayers`.</summary>
public sealed unsafe class Sd3Transformer : IDisposable
{
    private readonly Sd3Config _config;
    private readonly PatchEmbed _patchEmbed;
    private readonly JointBlock[] _blocks;
    private readonly Unpatchify _unpatchify;
    private int _disposed;

    // Timestep embedding: sinusoidal → Linear → SiLU → Linear
    private Tensor? _timestepLinear1Weight, _timestepLinear1Bias;
    private Tensor? _timestepLinear2Weight, _timestepLinear2Bias;

    // Pooled text projection: Linear → SiLU → Linear
    private Tensor? _textLinear1Weight, _textLinear1Bias;
    private Tensor? _textLinear2Weight, _textLinear2Bias;

    // Context embedder: Linear(joint_attention_dim, hidden_size)
    private Tensor? _contextEmbedWeight, _contextEmbedBias;

    // Final layer: AdaLN-Continuous (SiLU + Linear → shift/scale) + Linear projection
    private Tensor? _normOutLinearWeight, _normOutLinearBias;
    private Tensor? _projOutWeight, _projOutBias;

    /// <summary>Creates an SD3/SD3.5 transformer from configuration.</summary>
    public Sd3Transformer(Sd3Config config)
    {
        _config = config;

        _patchEmbed = new PatchEmbed(config.PatchSize, config.InChannels, config.HiddenSize);

        int ffDim = config.HiddenSize * 4;
        _blocks = new JointBlock[config.Depth];
        HashSet<int> dualLayers = config.DualAttentionLayers is null
            ? new HashSet<int>()
            : new HashSet<int>(config.DualAttentionLayers);

        for (int i = 0; i < config.Depth; i++)
        {
            bool isLastBlock = i == config.Depth - 1;
            bool useDual = dualLayers.Contains(i);
            _blocks[i] = new JointBlock(
                config.HiddenSize,
                config.NumHeads,
                ffDim,
                isPreOnly: isLastBlock,
                useQkNorm: config.UseQkNorm,
                useDualAttention: useDual,
                qkNormEps: config.QkNormEps);
        }

        _unpatchify = new Unpatchify(config.PatchSize, config.InChannels);
    }

    /// <summary>Loads all transformer weights from named tensors using HuggingFace diffusers naming.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _patchEmbed.LoadWeights(
            weights["pos_embed.proj.weight"],
            weights["pos_embed.proj.bias"],
            weights.ContainsKey("pos_embed.pos_embed") ? weights["pos_embed.pos_embed"] : null);

        _timestepLinear1Weight = weights["time_text_embed.timestep_embedder.linear_1.weight"];
        _timestepLinear1Bias = weights["time_text_embed.timestep_embedder.linear_1.bias"];
        _timestepLinear2Weight = weights["time_text_embed.timestep_embedder.linear_2.weight"];
        _timestepLinear2Bias = weights["time_text_embed.timestep_embedder.linear_2.bias"];

        _textLinear1Weight = weights["time_text_embed.text_embedder.linear_1.weight"];
        _textLinear1Bias = weights["time_text_embed.text_embedder.linear_1.bias"];
        _textLinear2Weight = weights["time_text_embed.text_embedder.linear_2.weight"];
        _textLinear2Bias = weights["time_text_embed.text_embedder.linear_2.bias"];

        _contextEmbedWeight = weights["context_embedder.weight"];
        _contextEmbedBias = weights["context_embedder.bias"];

        for (int i = 0; i < _config.Depth; i++)
            _blocks[i].LoadWeights(weights, $"transformer_blocks.{i}");

        _normOutLinearWeight = weights["norm_out.linear.weight"];
        _normOutLinearBias = weights["norm_out.linear.bias"];
        _projOutWeight = weights["proj_out.weight"];
        _projOutBias = weights["proj_out.bias"];
    }

    /// <summary>Yields every weight tensor for GPU preloading via <c>backend.PreloadWeights</c>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in EnumerateSharedWeights()) yield return w;

        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;
    }

    /// <summary>The number of streamable <see cref="JointBlock"/>s.</summary>
    public int BlockCount => _blocks.Length;

    /// <summary>The non-block weights: patch_embed, timestep/pooled-text embedders, context_embedder, and the
    /// final AdaLN-continuous + proj_out. Always resident on the block-range split's FIRST backend — see
    /// <see cref="ForwardSharded"/>.</summary>
    public IEnumerable<Tensor> EnumerateSharedWeights()
    {
        if (_timestepLinear1Weight is not null) yield return _timestepLinear1Weight;
        if (_timestepLinear1Bias is not null) yield return _timestepLinear1Bias;
        if (_timestepLinear2Weight is not null) yield return _timestepLinear2Weight;
        if (_timestepLinear2Bias is not null) yield return _timestepLinear2Bias;
        if (_textLinear1Weight is not null) yield return _textLinear1Weight;
        if (_textLinear1Bias is not null) yield return _textLinear1Bias;
        if (_textLinear2Weight is not null) yield return _textLinear2Weight;
        if (_textLinear2Bias is not null) yield return _textLinear2Bias;
        if (_contextEmbedWeight is not null) yield return _contextEmbedWeight;
        if (_contextEmbedBias is not null) yield return _contextEmbedBias;

        foreach (Tensor w in _patchEmbed.EnumerateWeights()) yield return w;

        if (_normOutLinearWeight is not null) yield return _normOutLinearWeight;
        if (_normOutLinearBias is not null) yield return _normOutLinearBias;
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
    }

    /// <summary>Enumerates only blocks <c>[startBlock, endBlock)</c>'s weights — the asymmetric preload a
    /// block-range shard needs (see <see cref="ForwardSharded"/>): the SECOND backend must load ONLY its block
    /// range, not <see cref="EnumerateWeights"/>'s full set, or VRAM pooling degrades into replication.</summary>
    public IEnumerable<Tensor> EnumerateBlockRangeWeights(int startBlock, int endBlock)
    {
        for (int i = startBlock; i < endBlock; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;
    }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="latent">Noisy latent [B, 16, H, W].</param>
    /// <param name="timestep">Timestep value (sigma * 1000 for flow matching).</param>
    /// <param name="context">Projected context embeddings [B, 154, hidden_size] (already through context_embedder).</param>
    /// <param name="pooled">Pooled text projection [B, 2048].</param>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor context, Tensor pooled)
    {
        int batch = (int)latent.Shape[0];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];

        Tensor imageTokens = _patchEmbed.Forward(backend, latent);
        int imgSeqLen = (int)imageTokens.Shape[1];
        (int gridH, int gridW) = _patchEmbed.GetGridSize(height, width);
        Sd3DebugDump.Dump("patch_embed", imageTokens);

        Tensor temb = ComputeTimestepEmbedding(backend, timestep, pooled, batch);
        Sd3DebugDump.Dump("time_text_embed", temb);

        Tensor currentImage = imageTokens;
        Tensor currentContext = context;

        for (int i = 0; i < _config.Depth; i++)
        {
            (Tensor newImage, Tensor newContext) = _blocks[i].Forward(backend, currentImage, currentContext, temb);

            if (!ReferenceEquals(currentImage, imageTokens))
                currentImage.Dispose();
            if (!ReferenceEquals(currentContext, context))
                currentContext.Dispose();

            currentImage = newImage;
            currentContext = newContext;

            Sd3DebugDump.Dump($"block_{i}_image", currentImage);
            // Last block (context_pre_only) returns context unchanged — skip the dump in that case
            // since the diffusers reference doesn't expose it (the hook records None).
            if (i < _config.Depth - 1)
                Sd3DebugDump.Dump($"block_{i}_context", currentContext);
        }

        if (!ReferenceEquals(currentContext, context))
            currentContext.Dispose();

        Tensor output = ApplyFinalLayer(backend, currentImage, temb, batch, imgSeqLen);
        Sd3DebugDump.Dump("proj_out", output);
        currentImage.Dispose();
        temb.Dispose();

        Tensor spatial = _unpatchify.Forward(output, batch, gridH, gridW);
        output.Dispose();
        Sd3DebugDump.DumpOutput(spatial);

        return spatial;
    }

    /// <summary>Predicts velocity for one denoising step with the <see cref="JointBlock"/> loop split across two
    /// backends: blocks <c>[0, splitBlock)</c> run on <paramref name="backendA"/>, blocks
    /// <c>[splitBlock, BlockCount)</c> on <paramref name="backendB"/> — a sequential pipeline split, not
    /// tensor-parallel, so there is <b>no latency win</b>; the win is VRAM pooling. <paramref name="backendA"/>
    /// must hold <see cref="EnumerateSharedWeights"/> + <see cref="EnumerateBlockRangeWeights"/><c>(0,
    /// splitBlock)</c>; <paramref name="backendB"/> must hold ONLY <see cref="EnumerateBlockRangeWeights"/>
    /// <c>(splitBlock, BlockCount)</c>. <paramref name="context"/>/<paramref name="pooled"/> are caller-owned
    /// (cached across steps by the pipeline) and are never disposed here, matching <see cref="Forward"/>.</summary>
    public Tensor ForwardSharded(IBackend backendA, IBackend backendB, Tensor latent, float timestep,
        Tensor context, Tensor pooled, int splitBlock)
    {
        if (splitBlock <= 0 || splitBlock >= _config.Depth)
            throw new ArgumentOutOfRangeException(nameof(splitBlock),
                $"splitBlock must be in (0, {_config.Depth}) — 0 or {_config.Depth} would put the whole MMDiT on one backend.");

        int batch = (int)latent.Shape[0];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];

        Tensor imageTokens = _patchEmbed.Forward(backendA, latent);
        int imgSeqLen = (int)imageTokens.Shape[1];
        (int gridH, int gridW) = _patchEmbed.GetGridSize(height, width);

        Tensor temb = ComputeTimestepEmbedding(backendA, timestep, pooled, batch);

        (Tensor imageA, Tensor contextA) = ForwardBlocksRange(backendA, imageTokens, context, temb, 0, splitBlock);
        // imageTokens is transformer-owned scratch (unlike context/pooled, which the pipeline reuses across
        // steps) — always superseded by a fresh tensor here (the image stream never has JointBlock's
        // context_pre_only passthrough), so it is always safe to free once the A-range block loop is done.
        imageTokens.Dispose();

        // ── hand off to backendB: image + context streams and temb (every block's AdaLN reads temb) ──
        Tensor imageB = new Tensor(imageA.Shape, imageA.DType);
        backendB.CopyFromPeer(imageB, imageA, backendA);
        imageA.Dispose();
        Tensor contextB = new Tensor(contextA.Shape, contextA.DType);
        backendB.CopyFromPeer(contextB, contextA, backendA);
        // contextA is always fresh too — A's range [0, splitBlock) never reaches the final context_pre_only
        // block (index Depth-1), so it can never alias the caller-owned `context`.
        contextA.Dispose();
        Tensor tembB = new Tensor(temb.Shape, temb.DType);
        backendB.CopyFromPeer(tembB, temb, backendA);

        (Tensor imageFinalB, Tensor contextFinalB) = ForwardBlocksRange(backendB, imageB, contextB, tembB, splitBlock, _config.Depth);
        imageB.Dispose();
        tembB.Dispose();
        // B's range always includes the final context_pre_only block, so contextFinalB may already be the
        // (disposed) product of ForwardBlocksRange's own internal passthrough guard, or still == contextB —
        // Dispose is idempotent (Interlocked.Exchange pattern), so both are freed safely either way.
        contextFinalB.Dispose();
        if (!ReferenceEquals(contextFinalB, contextB)) contextB.Dispose();

        // ── hand back to backendA: the final-layer weights live there (EnumerateSharedWeights) ──
        Tensor imageFinal = new Tensor(imageFinalB.Shape, imageFinalB.DType);
        backendA.CopyFromPeer(imageFinal, imageFinalB, backendB);
        imageFinalB.Dispose();

        Tensor output = ApplyFinalLayer(backendA, imageFinal, temb, batch, imgSeqLen);
        imageFinal.Dispose();
        temb.Dispose();

        Tensor spatial = _unpatchify.Forward(output, batch, gridH, gridW);
        output.Dispose();
        return spatial;
    }

    /// <summary>Runs <see cref="JointBlock"/>s <c>[startBlock, endBlock)</c> in order — the seam a block-range
    /// shard splits on (see <see cref="ForwardSharded"/>). Mirrors <see cref="Forward"/>'s own loop body exactly,
    /// windowed: never disposes <paramref name="image"/>/<paramref name="context"/> themselves (the caller owns
    /// them and decides when they're superseded), only intermediates created strictly within this range.</summary>
    private (Tensor image, Tensor context) ForwardBlocksRange(IBackend backend, Tensor image, Tensor context,
        Tensor temb, int startBlock, int endBlock)
    {
        Tensor currentImage = image;
        Tensor currentContext = context;
        for (int i = startBlock; i < endBlock; i++)
        {
            (Tensor newImage, Tensor newContext) = _blocks[i].Forward(backend, currentImage, currentContext, temb);
            if (!ReferenceEquals(currentImage, image)) currentImage.Dispose();
            if (!ReferenceEquals(currentContext, context)) currentContext.Dispose();
            currentImage = newImage;
            currentContext = newContext;
        }
        return (currentImage, currentContext);
    }

    /// <summary>Projects the combined context tensor from joint_attention_dim to hidden_size. Run once per prompt before <see cref="Forward"/>.</summary>
    public Tensor ProjectContext(IBackend backend, Tensor context)
    {
        int batch = (int)context.Shape[0];
        int seqLen = (int)context.Shape[1];

        TensorShape outShape = new TensorShape(batch, seqLen, _config.HiddenSize);
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Linear(output, context, _contextEmbedWeight!, _contextEmbedBias);

        return output;
    }

    private Tensor ComputeTimestepEmbedding(IBackend backend, float timestep, Tensor pooled, int batch)
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

        Tensor tEmb = new Tensor(hidShape, DType.F32);
        backend.Linear(tEmb, t1Activated, _timestepLinear2Weight!, _timestepLinear2Bias);
        t1Activated.Dispose();

        Tensor p1 = new Tensor(hidShape, DType.F32);
        backend.Linear(p1, pooled, _textLinear1Weight!, _textLinear1Bias);

        Tensor p1Activated = new Tensor(hidShape, DType.F32);
        backend.Silu(p1Activated, p1);
        p1.Dispose();

        Tensor pEmb = new Tensor(hidShape, DType.F32);
        backend.Linear(pEmb, p1Activated, _textLinear2Weight!, _textLinear2Bias);
        p1Activated.Dispose();

        Tensor temb = new Tensor(hidShape, DType.F32);
        backend.Add(temb, tEmb, pEmb);
        tEmb.Dispose();
        pEmb.Dispose();

        return temb;
    }

    /// <summary>Final AdaLN-Continuous: SiLU(temb) → Linear → [scale, shift] modulation, then unparameterized LayerNorm + modulate + proj_out.</summary>
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

        // Diffusers SD3 final-layer order: [shift, scale]
        // (mod[:dim] = shift, mod[dim:] = scale)
        float* modPtr = (float*)modParams.DataPointer;

        Tensor normed = new Tensor(hidShape, DType.F32);
        DiTUtils.LayerNormNoAffine(normed, hidden, batch, seqLen, dim);

        Tensor modulated = new Tensor(hidShape, DType.F32);
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
                    float shift = modPtr[modBase + d];
                    float scale = modPtr[modBase + dim + d];
                    outModPtr[vecOffset + d] = normPtr[vecOffset + d] * (1.0f + scale) + shift;
                }
            }
        }
        normed.Dispose();
        modParams.Dispose();
        Sd3DebugDump.Dump("norm_out", modulated);

        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor projected = new Tensor(outShape, DType.F32);
        backend.Linear(projected, modulated, _projOutWeight!, _projOutBias);
        modulated.Dispose();

        return projected;
    }

    /// <summary>Pipeline-level debug hook: dumps the post-denoise, pre-VAE latent under <c>$SD3_DEBUG_DIR/final_latent.bin</c> when the env var is set. Used by the layer-by-layer diff harness to capture pipeline state.</summary>
    public static void DumpFinalLatent(Tensor latent) => Sd3DebugDump.Dump("final_latent", latent);

    /// <summary>Releases all tensor references held by this transformer.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _timestepLinear1Weight = null; _timestepLinear1Bias = null;
            _timestepLinear2Weight = null; _timestepLinear2Bias = null;
            _textLinear1Weight = null; _textLinear1Bias = null;
            _textLinear2Weight = null; _textLinear2Bias = null;
            _contextEmbedWeight = null; _contextEmbedBias = null;
            _normOutLinearWeight = null; _normOutLinearBias = null;
            _projOutWeight = null; _projOutBias = null;
        }
        GC.SuppressFinalize(this);
    }
}
