using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Flux Diffusion Transformer. Processes packed latent image tokens and T5 text embeddings through double-stream (joint attention) and single-stream (parallel attention+MLP) blocks with RoPE positional encoding.</summary>
public sealed unsafe class FluxTransformer : IDisposable
{
    private readonly FluxConfig _config;
    private readonly FluxDoubleStreamBlock[] _doubleBlocks;
    private readonly FluxSingleStreamBlock[] _singleBlocks;
    private readonly FluxRope _rope;
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

    /// <summary>Creates a Flux transformer from configuration.</summary>
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
        // Image input projection (x_embedder in diffusers)
        _xEmbedWeight = weights["x_embedder.weight"];
        _xEmbedBias = weights["x_embedder.bias"];

        // Text input projection (context_embedder)
        _contextEmbedWeight = weights["context_embedder.weight"];
        _contextEmbedBias = weights["context_embedder.bias"];

        // Timestep embedding MLP
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

        // Double-stream blocks
        for (int i = 0; i < _config.Depth; i++)
            _doubleBlocks[i].LoadWeights(weights, $"transformer_blocks.{i}");

        // Single-stream blocks
        for (int i = 0; i < _config.DepthSingleBlocks; i++)
            _singleBlocks[i].LoadWeights(weights, $"single_transformer_blocks.{i}");

        // Final layer
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

    private static long SumBytes(IEnumerable<Tensor> tensors)
    {
        long total = 0;
        foreach (Tensor t in tensors) total += t.ElementCount * t.DType.SizeInBytes;
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
    /// <returns>Predicted velocity [B, imgSeqLen, 64].</returns>
    public Tensor Forward(IBackend backend, Tensor packedLatent, Tensor t5Embeddings, float sigma,
        Tensor clipPooled, float guidanceScale, int txtSeqLen, int hPacked, int wPacked)
    {
        int batch = (int)packedLatent.Shape[0];
        int imgSeqLen = (int)packedLatent.Shape[1];
        int totalSeqLen = txtSeqLen + imgSeqLen;
        int hidden = _config.HiddenSize;

        // ── 1. Project image tokens: [B, imgSeqLen, 64] → [B, imgSeqLen, 3072] ──
        TensorShape imgTokShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor imgTokens = new Tensor(imgTokShape, DType.F32);
        backend.Linear(imgTokens, packedLatent, _xEmbedWeight!, _xEmbedBias);

        // ── 2. Project text tokens: [B, txtSeqLen, 4096] → [B, txtSeqLen, 3072] ──
        TensorShape txtTokShape = new TensorShape(batch, txtSeqLen, hidden);
        Tensor txtTokens = new Tensor(txtTokShape, DType.F32);
        backend.Linear(txtTokens, t5Embeddings, _contextEmbedWeight!, _contextEmbedBias);

        // ── 3. Compute temb = timestep_embed + clip_pooled_embed + optional guidance_embed ──
        Tensor temb = ComputeTimestepEmbedding(backend, sigma, clipPooled, guidanceScale, batch);

        // ── 4. Precompute RoPE for this resolution ──
        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        _rope.Precompute(posIds);
        posIds.Dispose();

        // ── 5. Double-stream blocks ──
        Tensor currentImg = imgTokens;
        Tensor currentTxt = txtTokens;

        for (int i = 0; i < _config.Depth; i++)
        {
            BeforeBlockForward?.Invoke(i);
            (Tensor newImg, Tensor newTxt) = _doubleBlocks[i].Forward(backend, currentImg, currentTxt, temb, _rope);

            if (!ReferenceEquals(currentImg, imgTokens))
                currentImg.Dispose();
            if (!ReferenceEquals(currentTxt, txtTokens))
                currentTxt.Dispose();

            currentImg = newImg;
            currentTxt = newTxt;
        }

        // ── 6. Concatenate text + image for single-stream processing ──
        TensorShape concatShape = new TensorShape(batch, totalSeqLen, hidden);
        Tensor x = new Tensor(concatShape, DType.F32);
        ConcatAlongSeqDim3D(x, currentTxt, currentImg, batch, txtSeqLen, imgSeqLen, hidden);

        if (!ReferenceEquals(currentImg, imgTokens))
            currentImg.Dispose();
        if (!ReferenceEquals(currentTxt, txtTokens))
            currentTxt.Dispose();
        imgTokens.Dispose();
        txtTokens.Dispose();

        // ── 7. Single-stream blocks ──
        for (int i = 0; i < _config.DepthSingleBlocks; i++)
        {
            BeforeBlockForward?.Invoke(_config.Depth + i);
            Tensor newX = _singleBlocks[i].Forward(backend, x, temb, _rope);
            x.Dispose();
            x = newX;
        }

        // ── 8. Extract image tokens: discard text tokens ──
        TensorShape imgOutShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor imgOut = new Tensor(imgOutShape, DType.F32);
        ExtractImageTokens(imgOut, x, batch, txtSeqLen, imgSeqLen, hidden);
        x.Dispose();

        // ── 9. Final layer: AdaLN + proj_out ──
        Tensor output = ApplyFinalLayer(backend, imgOut, temb, batch, imgSeqLen);
        imgOut.Dispose();
        temb.Dispose();

        return output;
    }

    private Tensor ComputeTimestepEmbedding(IBackend backend, float sigma, Tensor clipPooled, float guidanceScale, int batch)
    {
        int hidden = _config.HiddenSize;

        // Sinusoidal timestep embedding: Flux uses time_factor=1000
        // sigma is in [0,1], scaled to [0, 1000] by the sinusoidal function
        float scaledTimestep = sigma * 1000.0f;

        TensorShape sinShape = new TensorShape(batch, 256);
        Tensor sinEmbed = new Tensor(sinShape, DType.F32);
        ComputeSinusoidalTimestep(sinEmbed, scaledTimestep, batch);

        // Timestep MLP: Linear(256, hidden) → SiLU → Linear(hidden, hidden)
        TensorShape hidShape = new TensorShape(batch, hidden);
        Tensor t1 = new Tensor(hidShape, DType.F32);
        backend.Linear(t1, sinEmbed, _timestepLinear1Weight!, _timestepLinear1Bias);
        sinEmbed.Dispose();

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

        // temb = timestep_emb + clip_emb
        Tensor temb = new Tensor(hidShape, DType.F32);
        backend.Add(temb, tEmb, pEmb);
        tEmb.Dispose();
        pEmb.Dispose();

        // Optional guidance embedding (Dev only)
        if (_config.GuidanceEmbed && _guidanceLinear1Weight != null)
        {
            // Embed guidance scale the same way as timestep
            Tensor gSin = new Tensor(sinShape, DType.F32);
            ComputeSinusoidalTimestep(gSin, guidanceScale * 1000.0f, batch);

            Tensor g1 = new Tensor(hidShape, DType.F32);
            backend.Linear(g1, gSin, _guidanceLinear1Weight!, _guidanceLinear1Bias);
            gSin.Dispose();

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

    /// <summary>Sinusoidal timestep embedding with flip_sin_to_cos=True. Output: [cos_0..cos_127, sin_0..sin_127].</summary>
    private static void ComputeSinusoidalTimestep(Tensor output, float timestep, int batch)
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

        // LayerNorm (no affine) + modulate
        TensorShape seqShape = new TensorShape(batch, seqLen, dim);
        Tensor normed = new Tensor(seqShape, DType.F32);
        LayerNormNoAffine(normed, hidden, batch, seqLen, dim);

        Tensor modulated = new Tensor(seqShape, DType.F32);
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
        modParams.Dispose();

        // Linear projection: [B, seqLen, hidden] → [B, seqLen, out_channels]
        TensorShape projShape = new TensorShape(batch, seqLen, outDim);
        Tensor projected = new Tensor(projShape, DType.F32);
        backend.Linear(projected, modulated, _projOutWeight!, _projOutBias);
        modulated.Dispose();

        return projected;
    }

    // ── Helper methods ──

    private static void LayerNormNoAffine(Tensor output, Tensor input, int batch, int seqLen, int dim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int offset = (b * seqLen + s) * dim;

                float mean = 0f;
                for (int d = 0; d < dim; d++)
                    mean += inPtr[offset + d];
                mean /= dim;

                float variance = 0f;
                for (int d = 0; d < dim; d++)
                {
                    float diff = inPtr[offset + d] - mean;
                    variance += diff * diff;
                }
                variance /= dim;

                float invStd = 1.0f / MathF.Sqrt(variance + 1e-6f);
                for (int d = 0; d < dim; d++)
                    outPtr[offset + d] = (inPtr[offset + d] - mean) * invStd;
            }
        }
    }

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
        }
        GC.SuppressFinalize(this);
    }
}
