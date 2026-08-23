using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Flux Diffusion Transformer. Processes packed latent image tokens and T5 text embeddings through double-stream (joint attention) and single-stream (parallel attention+MLP) blocks with RoPE positional encoding.</summary>
public sealed unsafe class FluxTransformer : IDisposable
{
    private readonly FluxConfig _config;
    private readonly FluxDoubleStreamBlock[] _doubleBlocks;
    private readonly FluxSingleStreamBlock[] _singleBlocks;
    private readonly FluxRope _rope;

    /// <summary>True when this instance runs the audited F16 block loop (HARTSY_DIT_F16): token streams are
    /// cast to F16 around the block loop and the residual stream rides at <see cref="ChromaF16.ResidualDamp"/>
    /// scale — the exact-damp recipe transplanted from Chroma (same Flux block architecture: every branch
    /// input passes a scale-invariant no-affine LayerNorm, and the final AdaLN-continuous norm cancels the
    /// factor before proj_out). Covers Dev / Schnell / Krea / Kontext (all Flux.1).</summary>
    private bool _f16Mode;

    // Rope-table signature: Precompute rebuilds the host trig tables AND re-uploads the GPU cos/sin tables —
    // it ran every forward even though positions only change with resolution / text length / Kontext ref grid.
    private long _ropeSig = long.MinValue;
    private int _disposed;

    // img_in: Linear(in_channels=64, hidden_size=3072)
    private Tensor? _xEmbedWeight, _xEmbedBias;

    // txt_in: Linear(context_in_dim=4096, hidden_size=3072)
    private Tensor? _contextEmbedWeight, _contextEmbedBias;

    // Timestep embedding MLP: sinusoidal → Linear → SiLU → Linear
    private Tensor? _timestepLinear1Weight, _timestepLinear1Bias;
    private Tensor? _timestepLinear2Weight, _timestepLinear2Bias;

    // Pooled text (CLIP) embedding MLP: Linear → SiLU → Linear
    private Tensor? _textLinear1Weight, _textLinear1Bias;
    private Tensor? _textLinear2Weight, _textLinear2Bias;

    // Optional guidance embedding MLP (Dev only): Linear → SiLU → Linear
    private Tensor? _guidanceLinear1Weight, _guidanceLinear1Bias;
    private Tensor? _guidanceLinear2Weight, _guidanceLinear2Bias;

    // Final layer: AdaLN-Continuous + proj_out
    private Tensor? _normOutLinearWeight, _normOutLinearBias;
    private Tensor? _projOutWeight, _projOutBias;

    public FluxTransformer(FluxConfig config)
    {
        _config = config;

        int mlpDim = (int)(config.HiddenSize * config.MlpRatio);

        _doubleBlocks = new FluxDoubleStreamBlock[config.Depth];
        for (int i = 0; i < config.Depth; i++)
        {
            _doubleBlocks[i] = new FluxDoubleStreamBlock(
                config.HiddenSize, config.NumHeads, mlpDim,
                config.QkvBias, config.QkNormEps);
        }

        _singleBlocks = new FluxSingleStreamBlock[config.DepthSingleBlocks];
        for (int i = 0; i < config.DepthSingleBlocks; i++)
        {
            _singleBlocks[i] = new FluxSingleStreamBlock(
                config.HiddenSize, config.NumHeads, mlpDim, config.QkNormEps);
        }

        _rope = new FluxRope(config.AxesDim, config.Theta);
    }

    /// <summary>Loads all transformer weights from named tensors using diffusers naming.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _f16Mode = DitDtype.Act == DType.F16;
        float branchDamp = _f16Mode ? ChromaF16.ResidualDamp : 1.0f;

        // Image input projection (x_embedder in diffusers)
        _xEmbedWeight = weights["x_embedder.weight"];
        _xEmbedBias = weights["x_embedder.bias"];

        // Text input projection (context_embedder)
        _contextEmbedWeight = weights["context_embedder.weight"];
        _contextEmbedBias = weights["context_embedder.bias"];

        if (_f16Mode)
        {
            // Enter the damped-residual regime at the embedders (see ChromaF16): both token streams start at
            // damp scale; the block-output damping keeps the whole residual stream there, and the final
            // no-affine LayerNorm cancels the factor exactly — proj_out is untouched.
            _xEmbedWeight.Fp8ScaleFactor *= ChromaF16.ResidualDamp;
            _xEmbedBias = ChromaF16.DampBias(_xEmbedBias!, ChromaF16.ResidualDamp);
            _contextEmbedWeight.Fp8ScaleFactor *= ChromaF16.ResidualDamp;
            _contextEmbedBias = ChromaF16.DampBias(_contextEmbedBias!, ChromaF16.ResidualDamp);
            Logs.Info(
                $"[Flux] F16 block loop active (residual damp 1/{1.0f / ChromaF16.ResidualDamp:F0})");
        }

        _timestepLinear1Weight = weights["time_text_embed.timestep_embedder.linear_1.weight"];
        _timestepLinear1Bias = weights["time_text_embed.timestep_embedder.linear_1.bias"];
        _timestepLinear2Weight = weights["time_text_embed.timestep_embedder.linear_2.weight"];
        _timestepLinear2Bias = weights["time_text_embed.timestep_embedder.linear_2.bias"];

        // Pooled text projection MLP
        _textLinear1Weight = weights["time_text_embed.text_embedder.linear_1.weight"];
        _textLinear1Bias = weights["time_text_embed.text_embedder.linear_1.bias"];
        _textLinear2Weight = weights["time_text_embed.text_embedder.linear_2.weight"];
        _textLinear2Bias = weights["time_text_embed.text_embedder.linear_2.bias"];

        // Optional guidance embedding (Dev only)
        if (_config.GuidanceEmbed)
        {
            _guidanceLinear1Weight = weights["time_text_embed.guidance_embedder.linear_1.weight"];
            _guidanceLinear1Bias = weights["time_text_embed.guidance_embedder.linear_1.bias"];
            _guidanceLinear2Weight = weights["time_text_embed.guidance_embedder.linear_2.weight"];
            _guidanceLinear2Bias = weights["time_text_embed.guidance_embedder.linear_2.bias"];
        }

        for (int i = 0; i < _config.Depth; i++)
            _doubleBlocks[i].LoadWeights(weights, $"transformer_blocks.{i}", branchDamp);

        for (int i = 0; i < _config.DepthSingleBlocks; i++)
            _singleBlocks[i].LoadWeights(weights, $"single_transformer_blocks.{i}", branchDamp);

        _normOutLinearWeight = weights["norm_out.linear.weight"];
        _normOutLinearBias = weights["norm_out.linear.bias"];
        _projOutWeight = weights["proj_out.weight"];
        _projOutBias = weights["proj_out.bias"];
    }

    /// <summary>Enumerates all weight tensors for GPU preloading. Equivalent to
    /// <see cref="EnumerateSharedWeights"/> followed by every block's weights — kept
    /// for callers that want the eager all-at-once preload pattern.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in EnumerateSharedWeights()) yield return w;
        for (int i = 0; i < BlockCount; i++)
        {
            foreach (Tensor w in GetBlock(i).EnumerateWeights()) yield return w;
        }
    }

    /// <summary>Enumerates the always-resident weights — input projections, timestep/text
    /// MLPs, optional guidance MLP, final AdaLN + projection. These are touched on every
    /// forward pass regardless of which block is currently executing, so the streaming
    /// controller does not manage them — callers should preload them eagerly via
    /// <see cref="IBackend.PreloadWeights"/> when streaming the per-block weights.
    /// Total: ~80 MB for Flux at fp8, negligible against the budget.</summary>
    public IEnumerable<Tensor> EnumerateSharedWeights()
    {
        if (_xEmbedWeight is not null) yield return _xEmbedWeight;
        if (_xEmbedBias is not null) yield return _xEmbedBias;
        if (_contextEmbedWeight is not null) yield return _contextEmbedWeight;
        if (_contextEmbedBias is not null) yield return _contextEmbedBias;
        if (_timestepLinear1Weight is not null) yield return _timestepLinear1Weight;
        if (_timestepLinear1Bias is not null) yield return _timestepLinear1Bias;
        if (_timestepLinear2Weight is not null) yield return _timestepLinear2Weight;
        if (_timestepLinear2Bias is not null) yield return _timestepLinear2Bias;
        if (_textLinear1Weight is not null) yield return _textLinear1Weight;
        if (_textLinear1Bias is not null) yield return _textLinear1Bias;
        if (_textLinear2Weight is not null) yield return _textLinear2Weight;
        if (_textLinear2Bias is not null) yield return _textLinear2Bias;
        if (_guidanceLinear1Weight is not null) yield return _guidanceLinear1Weight;
        if (_guidanceLinear1Bias is not null) yield return _guidanceLinear1Bias;
        if (_guidanceLinear2Weight is not null) yield return _guidanceLinear2Weight;
        if (_guidanceLinear2Bias is not null) yield return _guidanceLinear2Bias;
        if (_normOutLinearWeight is not null) yield return _normOutLinearWeight;
        if (_normOutLinearBias is not null) yield return _normOutLinearBias;
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
    }

    /// <summary>The number of streamable blocks: double-stream blocks first,
    /// then single-stream. Indexes <c>[0, Depth)</c> map to double blocks;
    /// <c>[Depth, Depth + DepthSingleBlocks)</c> to single blocks.</summary>
    public int BlockCount => _doubleBlocks.Length + _singleBlocks.Length;

    /// <summary>Input dimension of the <c>x_embedder</c> linear, derived from the loaded weight shape. Returns <c>64</c> for vanilla Flux (16 latent channels × 2×2 packing) or <c>128</c> for FLUX.1 Tools variants (Canny / Depth / Fill — 32 channels × 2×2). Pipelines use this to detect whether a control image must be concatenated to the noise latent before the transformer pass. Returns <c>0</c> when weights aren't yet loaded.</summary>
    public int XEmbedInputDim => _xEmbedWeight is not null ? (int)_xEmbedWeight.Shape[1] : 0;

    /// <summary>Returns the streamable block at the given index. Wrappers are
    /// instantiated on demand (cheap) — they hold a reference to the underlying
    /// double/single block and forward enumeration calls to it.</summary>
    public IStreamingBlock GetBlock(int idx)
    {
        if (idx < 0 || idx >= BlockCount) throw new ArgumentOutOfRangeException(nameof(idx));
        if (idx < _doubleBlocks.Length)
        {
            return new DoubleBlockHandle(_doubleBlocks[idx]);
        }
        return new SingleBlockHandle(_singleBlocks[idx - _doubleBlocks.Length]);
    }

    /// <summary>Optional hook invoked immediately before each block's forward pass
    /// during <see cref="Forward"/>. The block index passed in is the same one used
    /// by <see cref="GetBlock"/>. Pipelines plug a <see cref="BlockStreamingController"/>
    /// in here to drive prefetch and eviction; left null, the transformer behaves
    /// exactly as before (caller must have all weights resident).</summary>
    public Action<int>? BeforeBlockForward { get; set; }

    /// <summary>Wrapper around a <see cref="FluxDoubleStreamBlock"/> that satisfies
    /// <see cref="IStreamingBlock"/>. Pre-computes byte size at construction so the
    /// controller's budget heuristic doesn't pay enumeration cost on every query.</summary>
    private sealed class DoubleBlockHandle : IStreamingBlock
    {
        private readonly FluxDoubleStreamBlock _block;
        public DoubleBlockHandle(FluxDoubleStreamBlock block)
        {
            _block = block;
            EstimatedWeightBytes = SumBytes(block.EnumerateWeights());
        }
        public long EstimatedWeightBytes { get; }
        public IEnumerable<Tensor> EnumerateWeights() => _block.EnumerateWeights();
    }

    private sealed class SingleBlockHandle : IStreamingBlock
    {
        private readonly FluxSingleStreamBlock _block;
        public SingleBlockHandle(FluxSingleStreamBlock block)
        {
            _block = block;
            EstimatedWeightBytes = SumBytes(block.EnumerateWeights());
        }
        public long EstimatedWeightBytes { get; }
        public IEnumerable<Tensor> EnumerateWeights() => _block.EnumerateWeights();
    }

    /// <summary>Sums a block's on-device footprint, via <see cref="DType.ComputeByteCount"/> so block-quantized
    /// dtypes are measured by their super-block size rather than a per-element size they do not have.</summary>
    /// <remarks>The naive <c>ElementCount * SizeInBytes</c> this replaces silently reported <b>0 bytes</b> for every
    /// K-quant weight — <c>DType.Q4_K</c> is declared <c>("Q4_K", 0, true, 144, 256)</c>, i.e. `SizeInBytes` is 0 and
    /// the real footprint lives in `BlockByteSize`/`BlockElementCount`. A zero total made the resident-vs-streamed
    /// comparison trivially true, so **GGUF checkpoints always took the eager path and streaming never engaged** —
    /// the low-VRAM feature was inert for exactly the quantized models most likely to need it (e.g. flux2-dev-Q4_K_S).
    /// No-op for non-quantized dtypes.</remarks>
    private static long SumBytes(IEnumerable<Tensor> tensors)
    {
        long total = 0;
        foreach (Tensor t in tensors) total += t.DType.ComputeByteCount(t.ElementCount);
        return total;
    }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="packedLatent">Packed latent tokens [B, imgSeqLen, 64].</param>
    /// <param name="t5Embeddings">T5 text embeddings [B, txtSeqLen, 4096].</param>
    /// <param name="sigma">Current sigma (noise level, 0-1 range).</param>
    /// <param name="clipPooled">CLIP-L pooled embedding [B, 768].</param>
    /// <param name="guidanceScale">Guidance scale (Dev only, embedded via MLP). Ignored for Schnell.</param>
    /// <param name="txtSeqLen">Text sequence length.</param>
    /// <param name="hPacked">Packed image height (latent_h / 2).</param>
    /// <param name="wPacked">Packed image width (latent_w / 2).</param>
    /// <param name="controlNetResiduals">Optional Flux DiT ControlNet residuals for this step (see
    /// <see cref="Adapters.FluxControlNet"/>). Injected additively into the double-block image stream and the
    /// image rows of the single-block stream with diffusers' interval mapping. Requires batch 1 and no Kontext
    /// reference tokens; pipelines route ControlNet steps through the eager host path (never the step graph).</param>
    /// <returns>Predicted velocity [B, imgSeqLen, 64].</returns>
    public Tensor Forward(IBackend backend, Tensor packedLatent, Tensor t5Embeddings, float sigma,
        Tensor clipPooled, float guidanceScale, int txtSeqLen, int hPacked, int wPacked, Tensor? attnBias = null,
        int refSeqLen = 0, int refHPacked = 0, int refWPacked = 0,
        Adapters.FluxControlNetResiduals? controlNetResiduals = null, DeviceFeatureCache? stepCache = null)
    {
        Tensor tembOuter = ComputeTimestepEmbedding(backend, sigma, clipPooled, guidanceScale, (int)packedLatent.Shape[0]);
        Tensor velocity = ForwardWithTemb(backend, packedLatent, t5Embeddings, tembOuter, txtSeqLen, hPacked, wPacked,
            attnBias, refSeqLen, refHPacked, refWPacked, controlNetResiduals, stepCache);
        tembOuter.Dispose();
        return velocity;
    }

    /// <summary>DiT-sharded forward, v1 PLAIN PATH ONLY: batch 1, no ControlNet residuals, no Kontext reference
    /// tokens, no attention bias (regional/Redux). Runs the same activation dtype as <see cref="Forward"/>'s plain
    /// path (F16 under the default-ON HARTSY_DIT_F16, else F32) — required for same-device parity, and F16 halves
    /// the boundary-copy bytes. The FLAT block space is <c>[0, Depth)</c> double blocks then <c>[Depth, Depth+DepthSingleBlocks)</c>
    /// single blocks; blocks <c>[0, splitBlock)</c> run on <paramref name="backendA"/> (which owns the shared
    /// embed/head weights), the rest on <paramref name="backendB"/>. A double-region split crosses img+txt+temb;
    /// a single-region split crosses the concatenated x+temb. The pipeline is responsible for routing excluded
    /// feature combinations to the unsharded path — this method throws on them.</summary>
    public Tensor ForwardSharded(IBackend backendA, IBackend backendB, Tensor packedLatent, Tensor t5Embeddings,
        float sigma, Tensor clipPooled, float guidanceScale, int txtSeqLen, int hPacked, int wPacked, int splitBlock)
    {
        int totalBlocks = _config.Depth + _config.DepthSingleBlocks;
        if (splitBlock <= 0 || splitBlock >= totalBlocks)
            throw new ArgumentOutOfRangeException(nameof(splitBlock),
                $"splitBlock must be in (0, {totalBlocks}) exclusive, got {splitBlock}.");
        int batch = (int)packedLatent.Shape[0];
        if (batch != 1)
            throw new HartsyInferenceException("Flux DiT sharding v1 requires batch 1.");
        if (BeforeBlockForward is not null)
            throw new InvalidOperationException(
                "DiT sharding and block streaming don't compose — BeforeBlockForward must be null on the sharded path.");

        int imgSeqLen = (int)packedLatent.Shape[1];
        int totalSeqLen = txtSeqLen + imgSeqLen;
        int hidden = _config.HiddenSize;

        Tensor temb = ComputeTimestepEmbedding(backendA, sigma, clipPooled, guidanceScale, batch);
        Tensor img = new Tensor(new TensorShape(batch, imgSeqLen, hidden), DType.F32);
        backendA.Linear(img, packedLatent, _xEmbedWeight!, _xEmbedBias);
        Tensor txt = new Tensor(new TensorShape(batch, txtSeqLen, hidden), DType.F32);
        backendA.Linear(txt, t5Embeddings, _contextEmbedWeight!, _contextEmbedBias);
        // FluxRope keys its GPU cos/sin tables per backend, so one shared _rope serves both ranges.
        EnsureRope(txtSeqLen, hPacked, wPacked, 0, 0, 0);

        // Mirror Forward's F16 hot path exactly (HARTSY_DIT_F16 is default-ON, and the load-time weight damp is
        // baked either way): the sharded loop must run the SAME dtype as the unsharded one or same-device parity
        // is impossible. F16 also halves the boundary-copy bytes for free.
        if (_f16Mode)
        {
            Tensor imgF16 = new Tensor(img.Shape, DType.F16);
            backendA.CastToF16(imgF16, img);
            img.Dispose();
            img = imgF16;
            Tensor txtF16 = new Tensor(txt.Shape, DType.F16);
            backendA.CastToF16(txtF16, txt);
            txt.Dispose();
            txt = txtF16;
        }
        DType act = img.DType;

        int doubleSplit = Math.Min(splitBlock, _config.Depth);
        int singleSplit = Math.Max(0, splitBlock - _config.Depth);
        Tensor? tembB = null;

        for (int i = 0; i < doubleSplit; i++)
        {
            (Tensor newImg, Tensor newTxt) = _doubleBlocks[i].Forward(backendA, img, txt, temb, _rope, null);
            img.Dispose();
            txt.Dispose();
            img = newImg;
            txt = newTxt;
        }

        IBackend tailBackend = backendA;
        if (splitBlock < _config.Depth)
        {
            // Split inside the double region: BOTH live streams cross (plus temb, which every block reads).
            // temb is only COPIED — the head on A reads the original after B finishes.
            tembB = DiTUtils.CopyAcross(backendB, backendA, temb);
            img = DiTUtils.MoveAcross(backendB, backendA, img);
            txt = DiTUtils.MoveAcross(backendB, backendA, txt);
            tailBackend = backendB;
            for (int i = doubleSplit; i < _config.Depth; i++)
            {
                (Tensor newImg, Tensor newTxt) = _doubleBlocks[i].Forward(backendB, img, txt, tembB, _rope, null);
                img.Dispose();
                txt.Dispose();
                img = newImg;
                txt = newTxt;
            }
        }

        Tensor x = new Tensor(new TensorShape(batch, totalSeqLen, hidden), act);
        tailBackend.Concat(x, new Tensor[] { txt, img }, 1);
        img.Dispose();
        txt.Dispose();

        for (int i = 0; i < singleSplit; i++)
        {
            Tensor newX = _singleBlocks[i].Forward(backendA, x, temb, _rope, null);
            x.Dispose();
            x = newX;
        }
        if (splitBlock >= _config.Depth)
        {
            // Split inside the single region: one concatenated stream crosses.
            tembB = DiTUtils.CopyAcross(backendB, backendA, temb);
            x = DiTUtils.MoveAcross(backendB, backendA, x);
        }
        for (int i = singleSplit; i < _config.DepthSingleBlocks; i++)
        {
            Tensor newX = _singleBlocks[i].Forward(backendB, x, tembB!, _rope, null);
            x.Dispose();
            x = newX;
        }
        tembB?.Dispose();

        // Back to A: the head (final norm + proj_out) lives in the shared weights there.
        x = DiTUtils.MoveAcross(backendA, backendB, x);
        Tensor imgOut = new Tensor(new TensorShape(batch, imgSeqLen, hidden), act);
        backendA.SliceRows(imgOut, x, txtSeqLen);
        x.Dispose();
        if (imgOut.DType == DType.F16)
        {
            // Back to F32 for the final norm + proj_out (velocity precision matters across Euler steps).
            Tensor imgOutF32 = new Tensor(new TensorShape(batch, imgSeqLen, hidden), DType.F32);
            backendA.CastToF32(imgOutF32, imgOut);
            imgOut.Dispose();
            imgOut = imgOutF32;
        }
        Tensor output = ApplyFinalLayer(backendA, imgOut, temb, batch, imgSeqLen);
        imgOut.Dispose();
        temb.Dispose();
        return output;
    }

    /// <summary>Weights of flat-indexed blocks <c>[startBlock, endBlock)</c> (doubles then singles) — the
    /// asymmetric-preload primitive for DiT sharding; pair with <see cref="EnumerateSharedWeights"/> on backend A.</summary>
    public IEnumerable<Tensor> EnumerateBlockRangeWeights(int startBlock, int endBlock)
    {
        for (int i = startBlock; i < endBlock; i++)
        {
            foreach (Tensor w in GetBlock(i).EnumerateWeights()) yield return w;
        }
    }

    /// <summary>Forward body with a caller-owned temb (shared by the eager and step-graph paths — the graph
    /// path computes temb inside the capture from fixed device sin buffers). Does NOT dispose temb.</summary>
    /// <param name="stepCache">Optional across-step First-Block cache (docs/Research/STEP_ACCELERATION.md §2;
    /// same pattern as <see cref="Sd3Transformer.Forward"/>). Only engages when <paramref name="refSeqLen"/> is 0
    /// — a Kontext reference-token request has <c>noiseSeqLen != imgSeqLen</c>, so the cached anchor (block 0's
    /// full image-token output) and the eventual reconstruction target (the noise-only slice) would be different
    /// shapes; <see cref="DeviceFeatureCache.StoreResidual"/> would throw. Flux.1 is guidance-embedded (no CFG
    /// pair on this path — see the step-graph comment below), so unlike SD3/HiDream this needs only ONE cache
    /// instance, not one per stream.</param>
    private Tensor ForwardWithTemb(IBackend backend, Tensor packedLatent, Tensor t5Embeddings, Tensor temb,
        int txtSeqLen, int hPacked, int wPacked, Tensor? attnBias,
        int refSeqLen, int refHPacked, int refWPacked,
        Adapters.FluxControlNetResiduals? controlNetResiduals = null, DeviceFeatureCache? stepCache = null)
    {
        int batch = (int)packedLatent.Shape[0];
        int imgSeqLen = (int)packedLatent.Shape[1];
        int totalSeqLen = txtSeqLen + imgSeqLen;
        int hidden = _config.HiddenSize;
        // Kontext: packedLatent is [noise ; reference] concatenated along the sequence. The reference tokens
        // condition the model (channel-0=1 in RoPE) but are NOT denoised — the final output keeps only the
        // leading noise tokens. noiseSeqLen = imgSeqLen - refSeqLen.
        int noiseSeqLen = imgSeqLen - refSeqLen;

        // ── 1. Project image tokens: [B, imgSeqLen, 64] → [B, imgSeqLen, 3072] ──
        TensorShape imgTokShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor imgTokens = new Tensor(imgTokShape, DType.F32);
        backend.Linear(imgTokens, packedLatent, _xEmbedWeight!, _xEmbedBias);

        // ── 2. Project text tokens: [B, txtSeqLen, 4096] → [B, txtSeqLen, 3072] ──
        TensorShape txtTokShape = new TensorShape(batch, txtSeqLen, hidden);
        Tensor txtTokens = new Tensor(txtTokShape, DType.F32);
        backend.Linear(txtTokens, t5Embeddings, _contextEmbedWeight!, _contextEmbedBias);

        // ── 4. Precompute RoPE for this resolution — sig-cached (Precompute rebuilds the host trig
        //       tables AND re-uploads the GPU cos/sin tables; it ran EVERY forward before) ──
        EnsureRope(txtSeqLen, hPacked, wPacked, refSeqLen, refHPacked, refWPacked);

        // ── 4a. ControlNet residual prep: batch-1 / no-Kontext only (residuals are sized to the noise grid).
        //       Under the F16 weight-damp regime the token streams ride at ResidualDamp scale (damped embedder
        //       + branch-output weights — active even on the F32 loop below), so externally-computed residuals
        //       must be damped by the same factor to stay consistent; the final no-affine LayerNorm cancels it.
        //       Single-block residuals cover only the image rows — pad each with zero text rows ONCE so the
        //       per-block injection is a plain device Add (interval mapping reuses the same padded tensor). ──
        Tensor[]? cnDoubleResiduals = null;
        Tensor[]? cnSinglePadded = null;
        bool ownsCnDoubles = false;
        if (controlNetResiduals is not null)
        {
            if (batch != 1)
                throw new HartsyInferenceException("Flux ControlNet residual injection requires batch 1.");
            if (refSeqLen > 0)
                throw new HartsyInferenceException("Flux ControlNet residuals cannot be combined with Kontext reference tokens.");
            float residualDamp = _f16Mode ? ChromaF16.ResidualDamp : 1.0f;
            cnDoubleResiduals = controlNetResiduals.DoubleBlockResiduals;
            if (residualDamp != 1.0f && cnDoubleResiduals.Length > 0)
            {
                Tensor[] damped = new Tensor[cnDoubleResiduals.Length];
                for (int i = 0; i < cnDoubleResiduals.Length; i++)
                {
                    damped[i] = new Tensor(cnDoubleResiduals[i].Shape, DType.F32);
                    backend.Scale(damped[i], cnDoubleResiduals[i], residualDamp);
                }
                cnDoubleResiduals = damped;
                ownsCnDoubles = true;
            }
            if (controlNetResiduals.SingleBlockResiduals.Length > 0)
            {
                Tensor txtZeros = new Tensor(new TensorShape(1, txtSeqLen, hidden), DType.F32);
                txtZeros.AsSpan<float>().Clear();
                cnSinglePadded = new Tensor[controlNetResiduals.SingleBlockResiduals.Length];
                for (int i = 0; i < cnSinglePadded.Length; i++)
                {
                    Tensor res = controlNetResiduals.SingleBlockResiduals[i];
                    Tensor source = res;
                    if (residualDamp != 1.0f)
                    {
                        source = new Tensor(res.Shape, DType.F32);
                        backend.Scale(source, res, residualDamp);
                    }
                    cnSinglePadded[i] = new Tensor(new TensorShape(1, totalSeqLen, hidden), DType.F32);
                    backend.Concat(cnSinglePadded[i], new Tensor[] { txtZeros, source }, 1);
                    if (!ReferenceEquals(source, res)) source.Dispose();
                }
                txtZeros.Dispose();
            }
        }

        // ── 4b. F16 block loop (HARTSY_DIT_F16, B=1): one cast per stream before the loop — every block
        //       activation follows, halving the bandwidth-bound glue traffic and making all 57 SDPAs
        //       zero-cast cuDNN F16. Streams are already at ResidualDamp scale from the damped embedders.
        //       Cast back to F32 after the loop for the final norm (which cancels the damp) + Euler step.
        //       ControlNet steps stay F32 (residual Add needs matching dtype; CN steps are host-path anyway). ──
        bool f16Loop = _f16Mode && batch == 1 && controlNetResiduals is null;
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
        DType act = imgTokens.DType;

        // Only eligible when the anchor (block 0's full image-token output) and the eventual reconstruction
        // target (imgOut, the noise-only slice) are guaranteed the same shape/dtype, and no per-block side
        // effect (regional bias, ControlNet residual injection) would be silently skipped along with the
        // blocks themselves.
        bool cacheActive = stepCache is not null && refSeqLen == 0 && attnBias is null && controlNetResiduals is null;

        // ── 5. Double-stream blocks ──
        Tensor currentImg = imgTokens;
        Tensor currentTxt = txtTokens;
        Tensor? cacheAnchor = null;
        Tensor? hitReconstruction = null;

        int cnDoubleInterval = cnDoubleResiduals is { Length: > 0 }
            ? (int)Math.Ceiling((double)_config.Depth / cnDoubleResiduals.Length) : 0;
        for (int i = 0; i < _config.Depth; i++)
        {
            BeforeBlockForward?.Invoke(i);
            (Tensor newImg, Tensor newTxt) = _doubleBlocks[i].Forward(backend, currentImg, currentTxt, temb, _rope, attnBias);

            if (!ReferenceEquals(currentImg, imgTokens) && !ReferenceEquals(currentImg, cacheAnchor))
                currentImg.Dispose();
            if (!ReferenceEquals(currentTxt, txtTokens))
                currentTxt.Dispose();

            currentImg = newImg;
            currentTxt = newTxt;

            // ControlNet residual (diffusers interval mapping: base block i ← residual i / ceil(depth/count)).
            if (cnDoubleInterval > 0)
            {
                Tensor res = cnDoubleResiduals![Math.Min(i / cnDoubleInterval, cnDoubleResiduals.Length - 1)];
                Tensor injected = new Tensor(currentImg.Shape, currentImg.DType);
                backend.Add(injected, currentImg, res);
                currentImg.Dispose();
                currentImg = injected;
            }

            // Gate right after the very first double block: on a hit, skip straight past every remaining
            // double block, the img/txt concat, AND every single block — imgOut is the same shape at every
            // one of those points (guaranteed by cacheActive's refSeqLen==0 check above).
            if (i == 0 && cacheActive)
            {
                if (stepCache!.ShouldCompute(backend, currentImg))
                {
                    cacheAnchor = currentImg;
                }
                else
                {
                    hitReconstruction = stepCache.ApplyResidual(backend, currentImg);
                    currentImg.Dispose();
                    currentTxt.Dispose();
                    break;
                }
            }
        }
        bool cacheHit = hitReconstruction is not null;
        // Both branches have already moved past imgTokens/txtTokens by this point (the loop always runs at
        // least once before the cache gate can fire, hit or miss).
        if (!ReferenceEquals(currentImg, imgTokens)) imgTokens.Dispose();
        if (!ReferenceEquals(currentTxt, txtTokens)) txtTokens.Dispose();

        Tensor imgOut;
        if (cacheHit)
        {
            imgOut = hitReconstruction!;
            // Skipped the ControlNet single-block-residual list entirely (never entered), but cacheActive
            // already guarantees controlNetResiduals is null when a hit can happen — nothing to dispose.
        }
        else
        {
            // ── 6. Concatenate text + image for single-stream processing. B=1: device row-concat (the old
            //       host Buffer.MemoryCopy read both streams' DataPointer — a full pipeline drain per forward). ──
            TensorShape concatShape = new TensorShape(batch, totalSeqLen, hidden);
            Tensor x = new Tensor(concatShape, act);
            if (batch == 1)
                backend.Concat(x, new Tensor[] { currentTxt, currentImg }, 1);
            else
                ConcatAlongSeqDim3D(x, currentTxt, currentImg, batch, txtSeqLen, imgSeqLen, hidden);

            currentImg.Dispose();
            currentTxt.Dispose();

            // ── 7. Single-stream blocks ──
            int cnSingleInterval = cnSinglePadded is { Length: > 0 }
                ? (int)Math.Ceiling((double)_config.DepthSingleBlocks / cnSinglePadded.Length) : 0;
            for (int i = 0; i < _config.DepthSingleBlocks; i++)
            {
                BeforeBlockForward?.Invoke(_config.Depth + i);
                Tensor newX = _singleBlocks[i].Forward(backend, x, temb, _rope, attnBias);
                x.Dispose();
                x = newX;

                // ControlNet single-block residual: padded with zero text rows, so a plain Add touches only the
                // image rows (diffusers adds to the image hidden_states only).
                if (cnSingleInterval > 0)
                {
                    Tensor res = cnSinglePadded![Math.Min(i / cnSingleInterval, cnSinglePadded.Length - 1)];
                    Tensor injected = new Tensor(x.Shape, x.DType);
                    backend.Add(injected, x, res);
                    x.Dispose();
                    x = injected;
                }
            }

            // ControlNet scratch: the damped double-residual copies and the padded single residuals are
            // transformer-owned; the caller's original residual tensors are untouched.
            if (ownsCnDoubles)
            {
                foreach (Tensor t in cnDoubleResiduals!) t.Dispose();
            }
            if (cnSinglePadded is not null)
            {
                foreach (Tensor t in cnSinglePadded) t.Dispose();
            }

            // ── 8. Extract image tokens: discard text tokens (and, for Kontext, the trailing reference tokens).
            //        B=1: device row slice. ──
            TensorShape imgOutShape = new TensorShape(batch, noiseSeqLen, hidden);
            imgOut = new Tensor(imgOutShape, act);
            if (batch == 1)
                backend.SliceRows(imgOut, x, txtSeqLen);
            else
                ExtractImageTokens(imgOut, x, batch, txtSeqLen, noiseSeqLen, hidden);
            x.Dispose();

            if (cacheAnchor is not null)
            {
                stepCache!.StoreResidual(backend, cacheAnchor, imgOut);
                cacheAnchor.Dispose();
            }
        }

        if (imgOut.DType == DType.F16)
        {
            // Back to F32 for the final norm + proj_out (velocity precision matters across Euler steps).
            TensorShape imgOutShape = new TensorShape(batch, noiseSeqLen, hidden);
            Tensor imgOutF32 = new Tensor(imgOutShape, DType.F32);
            backend.CastToF32(imgOutF32, imgOut);
            imgOut.Dispose();
            imgOut = imgOutF32;
        }

        // ── 9. Final layer: AdaLN + proj_out ──
        Tensor output = ApplyFinalLayer(backend, imgOut, temb, batch, noiseSeqLen);
        imgOut.Dispose();

        return output;
    }

    // ── Persistent step-graph state (the Chroma round-3 recipe, single-forward variant: Flux.1 is
    // guidance-embedded, no CFG pair). The graph slot on the backend is SHARED across models; the graph is
    // per-(prompt refs, grid) signature and survives across generations as long as the pipeline never calls
    // FreeActivations. Pipeline MUST call InvalidateStepGraph before any FreeActivations / DiT FreeWeights. ──
    private const int GraphCaptureCall = 3;
    private Tensor? _latentFixed;
    private Tensor? _sinFixed;
    private Tensor? _guidanceSinFixed;
    private Tensor? _graphVelocity;
    private long _graphSig = long.MinValue;
    private int _graphSigCalls;
    private int _graphSigFlips;
    private bool _graphDead;

    /// <summary>Copies a freshly-Eulered packed latent into the transformer-owned FIXED latent buffer the
    /// captured graph reads, and returns that buffer — the pipeline denoise loop must feed the returned
    /// tensor to <see cref="ForwardGraphable"/> every step. A shape change resets the graph.</summary>
    public Tensor PrepareGraphLatent(IBackend backend, Tensor freshPackedLatent)
    {
        if (_latentFixed is not null && _latentFixed.Shape != freshPackedLatent.Shape)
        {
            InvalidateStepGraph(backend);
            _latentFixed.Dispose();
            _latentFixed = null;
            _graphVelocity?.Dispose();
            _graphVelocity = null;
        }
        _latentFixed ??= new Tensor(freshPackedLatent.Shape, DType.F32);
        backend.CopyInto(_latentFixed, freshPackedLatent);
        return _latentFixed;
    }

    /// <summary>Device copy of the fixed graph latent (for the pipeline's final read-back).</summary>
    public Tensor SnapshotGraphLatent(IBackend backend)
    {
        Tensor snap = new Tensor(_latentFixed!.Shape, DType.F32);
        backend.CopyInto(snap, _latentFixed);
        return snap;
    }

    /// <summary>Resets the backend graph slot and this transformer's signature so the next generation
    /// re-warms cleanly. Call before FreeActivations / FreeWeights.</summary>
    public void InvalidateStepGraph(IBackend backend)
    {
        backend.StepGraphReset();
        if (ReferenceEquals(backend.StepGraphOwner, this))
            backend.StepGraphOwner = null;
        _graphSig = long.MinValue;   // MinValue = "no sig": the next call resets WITHOUT counting a sig flip
        _graphSigCalls = 0;
    }

    /// <summary>Step-graph-aware forward. With <see cref="DitStepGraph"/> active and the latent routed
    /// through <see cref="PrepareGraphLatent"/>, the whole forward (temb MLPs → 57 blocks → final layer) is
    /// CUDA-graph-captured once per signature and replayed with one launch per step. The per-step timestep
    /// enters through a fixed device sin buffer refreshed OUTSIDE the capture; guidance sin + conditioning
    /// are step-invariant and pinned. <c>callerOwns</c> false = transformer-owned fixed velocity buffer,
    /// rewritten next step. Kontext (refSeqLen>0) and attnBias paths stay eager via <see cref="Forward"/>.</summary>
    public (Tensor velocity, bool callerOwns) ForwardGraphable(IBackend backend, Tensor packedLatent,
        Tensor t5Embeddings, float sigma, Tensor clipPooled, float guidanceScale,
        int txtSeqLen, int hPacked, int wPacked, DeviceFeatureCache? stepCache = null)
    {
        int batch = (int)packedLatent.Shape[0];
        // An armed step cache is per-step-variable topology (block count run varies with drift) — a captured
        // graph bakes one fixed sequence of launches and can't replay that (same exclusion as ZImageTransformer).
        bool graphMode = DitStepGraph.EnabledDefaultOn && backend.StepGraphSupported && !_graphDead
            && batch == 1 && ReferenceEquals(packedLatent, _latentFixed) && stepCache is null;
        if (!graphMode)
            return (Forward(backend, packedLatent, t5Embeddings, sigma, clipPooled, guidanceScale,
                txtSeqLen, hPacked, wPacked, stepCache: stepCache), true);

        // Refresh the fixed timestep-sin buffer — the ONLY per-step-varying content the captured graph reads.
        Tensor sinHost = new Tensor(new TensorShape(1, 256), DType.F32);
        ComputeSinusoidalTimestep(sinHost, sigma * 1000.0f, 1);
        _sinFixed ??= new Tensor(sinHost.Shape, DType.F32);
        backend.CopyInto(_sinFixed, sinHost);
        sinHost.Dispose();

        long sig = ((long)hPacked * 19349663L) ^ ((long)wPacked * 83492791L)
            ^ ((long)txtSeqLen * 2654435761L)
            ^ ((long)BitConverter.SingleToInt32Bits(guidanceScale) << 13)
            ^ ((long)System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(t5Embeddings) << 17)
            ^ ((long)System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(clipPooled) << 31);
        bool ownerLost = !ReferenceEquals(backend.StepGraphOwner, this);
        if (sig != _graphSig || ownerLost)
        {
            backend.StepGraphReset();
            if (!ownerLost && _graphSig != long.MinValue && ++_graphSigFlips > 8)
            {
                _graphDead = true;
                Logs.Warning("[Flux graph] signature flip storm — step-graph disabled for this session.");
            }
            _graphSig = sig;
            _graphSigCalls = 0;
            backend.StepGraphOwner = this;
        }
        else if (_graphSigCalls > GraphCaptureCall && !backend.StepGraphReady)
        {
            // Slot reset externally (FreeActivations elsewhere) while our signature stayed valid: re-warm.
            _graphSigCalls = 0;
        }
        _graphSigCalls++;
        if (_graphSigCalls == 1)
        {
            // Guidance sin is generation-invariant: refresh the fixed buffer once per signature.
            if (_config.GuidanceEmbed && _guidanceLinear1Weight != null)
            {
                Tensor gHost = new Tensor(new TensorShape(1, 256), DType.F32);
                ComputeSinusoidalTimestep(gHost, guidanceScale * 1000.0f, 1);
                _guidanceSinFixed ??= new Tensor(gHost.Shape, DType.F32);
                backend.CopyInto(_guidanceSinFixed, gHost);
                gHost.Dispose();
            }
            // Pin step-invariant conditioning as device-resident: pipelines host-materialize their prompt
            // caches, and a per-step PAGEABLE re-upload is an internally-syncing copy that breaks capture.
            backend.PreloadWeights(new List<Tensor> { t5Embeddings, clipPooled });
            EnsureRope(txtSeqLen, hPacked, wPacked, 0, 0, 0);
        }
        int outDim = _projOutWeight is not null ? (int)_projOutWeight.Shape[0] : 64;
        _graphVelocity ??= new Tensor(new TensorShape(1, (long)packedLatent.Shape[1], outDim), DType.F32);

        if (backend.StepGraphReady && _graphSigCalls > GraphCaptureCall)
        {
            backend.StepGraphLaunch();
            return (_graphVelocity!, false);
        }

        bool capture = _graphSigCalls == GraphCaptureCall;
        if (capture) backend.StepGraphBegin();
        try
        {
            RunStepIntoFixed(backend, packedLatent, t5Embeddings, clipPooled, txtSeqLen, hPacked, wPacked);
        }
        catch (Exception ex) when (capture)
        {
            backend.StepGraphReset();
            _graphDead = true;
            Logs.Warning($"[Flux graph] capture invalidated — falling back to eager: {ex}");
            RunStepIntoFixed(backend, packedLatent, t5Embeddings, clipPooled, txtSeqLen, hPacked, wPacked);
            return (_graphVelocity!, false);
        }
        if (capture)
        {
            try
            {
                backend.StepGraphEndAndLaunch();   // capture records without executing — this runs the step
                Logs.Info("[Flux graph] denoise step captured; replaying via cuGraphLaunch.");
            }
            catch (Exception ex)
            {
                backend.StepGraphReset();
                _graphDead = true;
                Logs.Warning($"[Flux graph] capture failed — falling back to eager: {ex.Message}");
                RunStepIntoFixed(backend, packedLatent, t5Embeddings, clipPooled, txtSeqLen, hPacked, wPacked);
            }
        }
        return (_graphVelocity!, false);
    }

    /// <summary>The captured (or capture-warming) step body: temb from the fixed device sin buffers, then
    /// the full forward, landing the velocity in the fixed pre-capture buffer via <c>CopyInto</c>.</summary>
    private void RunStepIntoFixed(IBackend backend, Tensor packedLatent, Tensor t5Embeddings,
        Tensor clipPooled, int txtSeqLen, int hPacked, int wPacked)
    {
        Tensor temb = ComputeTembFromSin(backend, _sinFixed!, _guidanceSinFixed, clipPooled, 1);
        Tensor v = ForwardWithTemb(backend, packedLatent, t5Embeddings, temb, txtSeqLen, hPacked, wPacked,
            null, 0, 0, 0);
        backend.CopyInto(_graphVelocity!, v);
        v.Dispose();
        temb.Dispose();
    }

    /// <summary>Precomputes the rope tables only when the (text len, grid, Kontext ref grid) signature changes.</summary>
    private void EnsureRope(int txtSeqLen, int hPacked, int wPacked, int refSeqLen, int refHPacked, int refWPacked)
    {
        long sig = ((long)txtSeqLen << 48) ^ ((long)hPacked << 36) ^ ((long)wPacked << 24)
            ^ ((long)refSeqLen << 16) ^ ((long)refHPacked << 8) ^ (long)refWPacked ^ 0x5F0F5F0F;
        if (sig == _ropeSig)
            return;
        Tensor posIds = refSeqLen > 0
            ? FluxRope.BuildPositionIdsKontext(txtSeqLen, hPacked, wPacked, refHPacked, refWPacked)
            : FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        _rope.Precompute(posIds);
        posIds.Dispose();
        _ropeSig = sig;
    }

    private Tensor ComputeTimestepEmbedding(IBackend backend, float sigma, Tensor clipPooled, float guidanceScale, int batch)
    {
        // Sinusoidal timestep embedding: Flux uses time_factor=1000; sigma in [0,1] scales to [0,1000].
        TensorShape sinShape = new TensorShape(batch, 256);
        Tensor sinEmbed = new Tensor(sinShape, DType.F32);
        ComputeSinusoidalTimestep(sinEmbed, sigma * 1000.0f, batch);

        Tensor? guidanceSin = null;
        if (_config.GuidanceEmbed && _guidanceLinear1Weight != null)
        {
            guidanceSin = new Tensor(sinShape, DType.F32);
            ComputeSinusoidalTimestep(guidanceSin, guidanceScale * 1000.0f, batch);
        }

        Tensor temb = ComputeTembFromSin(backend, sinEmbed, guidanceSin, clipPooled, batch);
        sinEmbed.Dispose();
        guidanceSin?.Dispose();
        return temb;
    }

    /// <summary>temb MLP core from (already-built) sinusoidal embeddings: timestep MLP + pooled-CLIP MLP
    /// (+ optional guidance MLP), all device ops — the step-graph path feeds FIXED device sin buffers so
    /// the whole temb compute is capturable.</summary>
    private Tensor ComputeTembFromSin(IBackend backend, Tensor sinEmbed, Tensor? guidanceSin, Tensor clipPooled, int batch)
    {
        int hidden = _config.HiddenSize;
        TensorShape hidShape = new TensorShape(batch, hidden);

        // Timestep MLP: Linear(256, hidden) → SiLU → Linear(hidden, hidden)
        Tensor t1 = new Tensor(hidShape, DType.F32);
        backend.Linear(t1, sinEmbed, _timestepLinear1Weight!, _timestepLinear1Bias);
        Tensor t1Act = new Tensor(hidShape, DType.F32);
        backend.Silu(t1Act, t1);
        t1.Dispose();
        Tensor tEmb = new Tensor(hidShape, DType.F32);
        backend.Linear(tEmb, t1Act, _timestepLinear2Weight!, _timestepLinear2Bias);
        t1Act.Dispose();

        // Pooled text (CLIP) MLP: Linear(768, hidden) → SiLU → Linear(hidden, hidden)
        Tensor p1 = new Tensor(hidShape, DType.F32);
        backend.Linear(p1, clipPooled, _textLinear1Weight!, _textLinear1Bias);
        Tensor p1Act = new Tensor(hidShape, DType.F32);
        backend.Silu(p1Act, p1);
        p1.Dispose();
        Tensor pEmb = new Tensor(hidShape, DType.F32);
        backend.Linear(pEmb, p1Act, _textLinear2Weight!, _textLinear2Bias);
        p1Act.Dispose();

        Tensor temb = new Tensor(hidShape, DType.F32);
        backend.Add(temb, tEmb, pEmb);
        tEmb.Dispose();
        pEmb.Dispose();

        if (guidanceSin is not null && _guidanceLinear1Weight != null)
        {
            Tensor g1 = new Tensor(hidShape, DType.F32);
            backend.Linear(g1, guidanceSin, _guidanceLinear1Weight!, _guidanceLinear1Bias);
            Tensor g1Act = new Tensor(hidShape, DType.F32);
            backend.Silu(g1Act, g1);
            g1.Dispose();
            Tensor gEmb = new Tensor(hidShape, DType.F32);
            backend.Linear(gEmb, g1Act, _guidanceLinear2Weight!, _guidanceLinear2Bias);
            g1Act.Dispose();

            Tensor tembNew = new Tensor(hidShape, DType.F32);
            backend.Add(tembNew, temb, gEmb);
            temb.Dispose();
            gEmb.Dispose();
            temb = tembNew;
        }

        return temb;
    }

    /// <summary>Sinusoidal timestep embedding with flip_sin_to_cos=True. Output: [cos_0..cos_127, sin_0..sin_127].
    /// Internal so <see cref="Adapters.FluxControlNet"/> (which mirrors this transformer's time_text_embed) reuses
    /// the exact same embedding.</summary>
    internal static void ComputeSinusoidalTimestep(Tensor output, float timestep, int batch)
    {
        float* outPtr = (float*)output.DataPointer;
        int halfDim = 128;

        for (int b = 0; b < batch; b++)
        {
            int baseOffset = b * 256;
            for (int i = 0; i < halfDim; i++)
            {
                float freq = MathF.Exp(-MathF.Log(10000.0f) * i / halfDim);
                float angle = timestep * freq;
                outPtr[baseOffset + i] = MathF.Cos(angle);
                outPtr[baseOffset + halfDim + i] = MathF.Sin(angle);
            }
        }
    }

    private Tensor ApplyFinalLayer(IBackend backend, Tensor hidden, Tensor temb, int batch, int seqLen)
    {
        int dim = _config.HiddenSize;
        int outDim = _config.OutChannels;

        // AdaLN-Continuous: SiLU(temb) → Linear → [scale, shift]
        TensorShape tembShape = new TensorShape(batch, dim);
        Tensor activated = new Tensor(tembShape, DType.F32);
        backend.Silu(activated, temb);

        TensorShape modShape = new TensorShape(batch, dim * 2);
        Tensor modParams = new Tensor(modShape, DType.F32);
        backend.Linear(modParams, activated, _normOutLinearWeight!, _normOutLinearBias);
        activated.Dispose();

        // LayerNorm (no affine) + modulate. B=1: device ops (the old host loops read the device-produced
        // modParams + block output — a full pipeline drain per forward, and capture-illegal under the step
        // graph). Chunk order is [scale, shift] (diffusers AdaLayerNormContinuous).
        TensorShape seqShape = new TensorShape(batch, seqLen, dim);
        Tensor modulated;
        if (batch == 1)
        {
            Tensor normed = new Tensor(seqShape, DType.F32);
            backend.LayerNormNoAffine(normed, hidden, 1e-6f);

            Tensor scaleRow = new Tensor(new TensorShape(1, dim), DType.F32);
            backend.SliceRows(scaleRow, modParams, 0);
            Tensor shiftRow = new Tensor(new TensorShape(1, dim), DType.F32);
            backend.SliceRows(shiftRow, modParams, 1);
            Tensor scalePlus1 = new Tensor(new TensorShape(1, dim), DType.F32);
            backend.AddScalar(scalePlus1, scaleRow, 1.0f);
            scaleRow.Dispose();

            modulated = new Tensor(seqShape, DType.F32);
            backend.AffineBroadcastLastDim(modulated, normed, scalePlus1, shiftRow);
            normed.Dispose();
            scalePlus1.Dispose();
            shiftRow.Dispose();
        }
        else
        {
            Tensor normed = new Tensor(seqShape, DType.F32);
            DiTUtils.LayerNormNoAffine(normed, hidden, batch, seqLen, dim);

            modulated = new Tensor(seqShape, DType.F32);
            float* normPtr = (float*)normed.DataPointer;
            float* modPtr = (float*)modParams.DataPointer;
            float* outModPtr = (float*)modulated.DataPointer;

            for (int b = 0; b < batch; b++)
            {
                int modBase = b * dim * 2;
                for (int s = 0; s < seqLen; s++)
                {
                    int vecOffset = (b * seqLen + s) * dim;
                    for (int d = 0; d < dim; d++)
                    {
                        float scale = modPtr[modBase + d];
                        float shift = modPtr[modBase + dim + d];
                        outModPtr[vecOffset + d] = normPtr[vecOffset + d] * (1.0f + scale) + shift;
                    }
                }
            }
            normed.Dispose();
        }
        modParams.Dispose();

        // Linear projection: [B, seqLen, hidden] → [B, seqLen, out_channels]
        TensorShape projShape = new TensorShape(batch, seqLen, outDim);
        Tensor projected = new Tensor(projShape, DType.F32);
        backend.Linear(projected, modulated, _projOutWeight!, _projOutBias);
        modulated.Dispose();

        return projected;
    }

    // ── Helper methods ──

    /// <summary>Concatenates two 3D tensors along the sequence dimension: [B,S1,D] + [B,S2,D] → [B,S1+S2,D].</summary>
    private static void ConcatAlongSeqDim3D(Tensor output, Tensor first, Tensor second,
        int batch, int firstSeqLen, int secondSeqLen, int dim)
    {
        float* firstPtr = (float*)first.DataPointer;
        float* secondPtr = (float*)second.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int totalSeqLen = firstSeqLen + secondSeqLen;

        for (int b = 0; b < batch; b++)
        {
            long firstBytes = (long)firstSeqLen * dim * sizeof(float);
            long secondBytes = (long)secondSeqLen * dim * sizeof(float);

            Buffer.MemoryCopy(
                firstPtr + b * firstSeqLen * dim,
                outPtr + b * totalSeqLen * dim,
                firstBytes, firstBytes);

            Buffer.MemoryCopy(
                secondPtr + b * secondSeqLen * dim,
                outPtr + b * totalSeqLen * dim + firstSeqLen * dim,
                secondBytes, secondBytes);
        }
    }

    /// <summary>Extracts image tokens from concatenated [text, image] sequence.</summary>
    private static void ExtractImageTokens(Tensor output, Tensor input, int batch, int txtSeqLen, int imgSeqLen, int dim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int totalSeqLen = txtSeqLen + imgSeqLen;

        for (int b = 0; b < batch; b++)
        {
            long imgBytes = (long)imgSeqLen * dim * sizeof(float);
            Buffer.MemoryCopy(
                inPtr + b * totalSeqLen * dim + txtSeqLen * dim,
                outPtr + b * imgSeqLen * dim,
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
            _timestepLinear1Weight = null;
            _timestepLinear1Bias = null;
            _timestepLinear2Weight = null;
            _timestepLinear2Bias = null;
            _textLinear1Weight = null;
            _textLinear1Bias = null;
            _textLinear2Weight = null;
            _textLinear2Bias = null;
            _guidanceLinear1Weight = null;
            _guidanceLinear1Bias = null;
            _guidanceLinear2Weight = null;
            _guidanceLinear2Bias = null;
            _normOutLinearWeight = null;
            _normOutLinearBias = null;
            _projOutWeight = null;
            _projOutBias = null;
            _latentFixed?.Dispose();
            _latentFixed = null;
            _sinFixed?.Dispose();
            _sinFixed = null;
            _guidanceSinFixed?.Dispose();
            _guidanceSinFixed = null;
            _graphVelocity?.Dispose();
            _graphVelocity = null;
        }
        GC.SuppressFinalize(this);
    }
}
