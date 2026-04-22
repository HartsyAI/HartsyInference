using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>SD3 Multi-Modal Diffusion Transformer (MMDiT). Jointly processes image patches and text tokens through symmetric dual-stream attention blocks. Supports both SD3 (MMDiT) and SD3.5 (MMDiT-X with dual attention).</summary>
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

    /// <summary>Creates an SD3 transformer from configuration.</summary>
    public Sd3Transformer(Sd3Config config)
    {
        _config = config;

        _patchEmbed = new PatchEmbed(config.PatchSize, config.InChannels, config.HiddenSize);

        int ffDim = config.HiddenSize * 4;
        _blocks = new JointBlock[config.Depth];
        for (int i = 0; i < config.Depth; i++)
        {
            bool isLastBlock = i == config.Depth - 1;
            _blocks[i] = new JointBlock(
                config.HiddenSize,
                config.NumHeads,
                ffDim,
                isPreOnly: isLastBlock,
                useQkNorm: config.UseQkNorm,
                qkNormEps: config.QkNormEps);
        }

        _unpatchify = new Unpatchify(config.PatchSize, config.InChannels);
    }

    /// <summary>Loads all transformer weights from named tensors using HuggingFace diffusers naming.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        // Patch embedding
        _patchEmbed.LoadWeights(
            weights["pos_embed.proj.weight"],
            weights["pos_embed.proj.bias"],
            weights.ContainsKey("pos_embed.pos_embed") ? weights["pos_embed.pos_embed"] : null);

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

        // Context embedder
        _contextEmbedWeight = weights["context_embedder.weight"];
        _contextEmbedBias = weights["context_embedder.bias"];

        // Transformer blocks
        for (int i = 0; i < _config.Depth; i++)
        {
            _blocks[i].LoadWeights(weights, $"transformer_blocks.{i}");
        }

        // Final layer
        _normOutLinearWeight = weights["norm_out.linear.weight"];
        _normOutLinearBias = weights["norm_out.linear.bias"];
        _projOutWeight = weights["proj_out.weight"];
        _projOutBias = weights["proj_out.bias"];
    }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="latent">Noisy latent [B, 16, H, W].</param>
    /// <param name="timestep">Timestep value (sigma * 1000 for flow matching).</param>
    /// <param name="context">Projected context embeddings [B, 154, hidden_size] (already through context_embedder).</param>
    /// <param name="pooled">Pooled text projection [B, 2048].</param>
    /// <returns>Predicted velocity [B, 16, H, W].</returns>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor context, Tensor pooled)
    {
        int batch = (int)latent.Shape[0];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];

        // ── 1. Patch embed latent → image tokens ────────────────────────
        Tensor imageTokens = _patchEmbed.Forward(backend, latent);
        int imgSeqLen = (int)imageTokens.Shape[1];
        (int gridH, int gridW) = _patchEmbed.GetGridSize(height, width);

        // ── 2. Timestep + pooled embedding → temb ───────────────────────
        Tensor temb = ComputeTimestepEmbedding(backend, timestep, pooled, batch);

        // ── 3. Run all JointBlocks ──────────────────────────────────────
        Tensor currentImage = imageTokens;
        Tensor currentContext = context;

        for (int i = 0; i < _config.Depth; i++)
        {
            (Tensor newImage, Tensor newContext) = _blocks[i].Forward(backend, currentImage, currentContext, temb);

            if (!ReferenceEquals(currentImage, imageTokens))
                currentImage.Dispose();
            if (!ReferenceEquals(currentContext, context) && i > 0)
                currentContext.Dispose();

            currentImage = newImage;
            currentContext = newContext;
        }

        // Dispose final context if it's not the original
        if (!ReferenceEquals(currentContext, context))
            currentContext.Dispose();

        // ── 4. Final layer: AdaLN-Continuous + Linear projection ────────
        Tensor output = ApplyFinalLayerWithTemb(backend, currentImage, temb, batch, imgSeqLen);
        currentImage.Dispose();
        temb.Dispose();

        // ── 5. Unpatchify → [B, 16, H, W] ──────────────────────────────
        Tensor spatial = _unpatchify.Forward(output, batch, gridH, gridW);
        output.Dispose();

        return spatial;
    }

    /// <summary>Projects the combined context tensor from joint_attention_dim to hidden_size. Call this before Forward() to prepare context.</summary>
    public Tensor ProjectContext(IBackend backend, Tensor context)
    {
        int batch = (int)context.Shape[0];
        int seqLen = (int)context.Shape[1];
        int inDim = (int)context.Shape[2];

        TensorShape outShape = new TensorShape(batch, seqLen, _config.HiddenSize);
        Tensor output = new Tensor(outShape, DType.F32);

        LinearProjectBatched(output, context, _contextEmbedWeight!, _contextEmbedBias!, batch, seqLen, inDim, _config.HiddenSize);

        return output;
    }

    private Tensor ComputeTimestepEmbedding(IBackend backend, float timestep, Tensor pooled, int batch)
    {
        int hidden = _config.HiddenSize;

        // Sinusoidal timestep embedding with flip_sin_to_cos=True (SD3 convention)
        // SD1.5 trap #5: SD3 uses [cos, sin] order (flip_sin_to_cos=True)
        TensorShape sinShape = new TensorShape(batch, 256);
        Tensor sinEmbed = new Tensor(sinShape, DType.F32);
        ComputeSinusoidalTimestep(sinEmbed, timestep, batch);

        // MLP: Linear(256, hidden) → SiLU → Linear(hidden, hidden)
        TensorShape hidShape = new TensorShape(batch, hidden);
        Tensor t1 = new Tensor(hidShape, DType.F32);
        LinearProject1D(t1, sinEmbed, _timestepLinear1Weight!, _timestepLinear1Bias!, batch, 256, hidden);
        sinEmbed.Dispose();

        Tensor t1Activated = new Tensor(hidShape, DType.F32);
        backend.Silu(t1Activated, t1);
        t1.Dispose();

        Tensor tEmb = new Tensor(hidShape, DType.F32);
        LinearProject1D(tEmb, t1Activated, _timestepLinear2Weight!, _timestepLinear2Bias!, batch, hidden, hidden);
        t1Activated.Dispose();

        // Pooled text projection: Linear(2048, hidden) → SiLU → Linear(hidden, hidden)
        Tensor p1 = new Tensor(hidShape, DType.F32);
        LinearProject1D(p1, pooled, _textLinear1Weight!, _textLinear1Bias!, batch, _config.PooledProjectionDim, hidden);

        Tensor p1Activated = new Tensor(hidShape, DType.F32);
        backend.Silu(p1Activated, p1);
        p1.Dispose();

        Tensor pEmb = new Tensor(hidShape, DType.F32);
        LinearProject1D(pEmb, p1Activated, _textLinear2Weight!, _textLinear2Bias!, batch, hidden, hidden);
        p1Activated.Dispose();

        // Combine: temb = timestep_emb + pooled_emb
        Tensor temb = new Tensor(hidShape, DType.F32);
        backend.Add(temb, tEmb, pEmb);
        tEmb.Dispose();
        pEmb.Dispose();

        return temb;
    }

    /// <summary>Sinusoidal timestep embedding with flip_sin_to_cos=True. Output: [cos_0..cos_127, sin_0..sin_127].</summary>
    private static void ComputeSinusoidalTimestep(Tensor output, float timestep, int batch)
    {
        float* outPtr = (float*)output.DataPointer;
        int halfDim = 128; // 256 / 2

        for (int b = 0; b < batch; b++)
        {
            int baseOffset = b * 256;
            for (int i = 0; i < halfDim; i++)
            {
                float freq = MathF.Exp(-MathF.Log(10000.0f) * i / halfDim);
                float angle = timestep * freq;
                // flip_sin_to_cos=True: output is [cos, sin]
                outPtr[baseOffset + i] = MathF.Cos(angle);
                outPtr[baseOffset + halfDim + i] = MathF.Sin(angle);
            }
        }
    }

    /// <summary>Applies the final AdaLN-Continuous + Linear projection. Takes the combined timestep+pooled embedding.</summary>
    private Tensor ApplyFinalLayerWithTemb(IBackend backend, Tensor hidden, Tensor temb, int batch, int seqLen)
    {
        int dim = _config.HiddenSize;
        int outDim = _config.PatchSize * _config.PatchSize * _config.InChannels;
        TensorShape hidShape = new TensorShape(batch, seqLen, dim);

        // AdaLN-Continuous: SiLU(temb) → Linear → [scale, shift]
        TensorShape tembShape = new TensorShape(batch, dim);
        Tensor activated = new Tensor(tembShape, DType.F32);
        backend.Silu(activated, temb);

        TensorShape modParamShape = new TensorShape(batch, dim * 2);
        Tensor modParams = new Tensor(modParamShape, DType.F32);
        LinearProject1D(modParams, activated, _normOutLinearWeight!, _normOutLinearBias!, batch, dim, dim * 2);
        activated.Dispose();

        // Split into scale and shift (each [B, dim])
        float* modPtr = (float*)modParams.DataPointer;

        // Unparameterized LayerNorm + modulate
        Tensor normed = new Tensor(hidShape, DType.F32);
        LayerNormNoAffine(normed, hidden, batch, seqLen, dim);

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
                    float scale = modPtr[modBase + d];
                    float shift = modPtr[modBase + dim + d];
                    outModPtr[vecOffset + d] = normPtr[vecOffset + d] * (1.0f + scale) + shift;
                }
            }
        }
        normed.Dispose();
        modParams.Dispose();

        // Final linear projection: [B, seqLen, hidden] → [B, seqLen, patch^2 * channels]
        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor projected = new Tensor(outShape, DType.F32);
        LinearProjectBatched(projected, modulated, _projOutWeight!, _projOutBias!, batch, seqLen, dim, outDim);
        modulated.Dispose();

        return projected;
    }

    /// <summary>Unparameterized LayerNorm.</summary>
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

    /// <summary>Linear projection for 1D vectors: output = input @ weight^T + bias. Input: [B, inDim], Output: [B, outDim].</summary>
    private static void LinearProject1D(Tensor output, Tensor input, Tensor weight, Tensor bias, int batch, int inDim, int outDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* wPtr = (float*)weight.DataPointer;
        float* bPtr = (float*)bias.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int inOffset = b * inDim;
            int outOffset = b * outDim;
            for (int o = 0; o < outDim; o++)
            {
                float sum = bPtr[o];
                int wOffset = o * inDim;
                for (int i = 0; i < inDim; i++)
                {
                    sum += inPtr[inOffset + i] * wPtr[wOffset + i];
                }
                outPtr[outOffset + o] = sum;
            }
        }
    }

    /// <summary>Batched linear projection: output = input @ weight^T + bias. Input: [B, S, inDim], Output: [B, S, outDim].</summary>
    private static void LinearProjectBatched(Tensor output, Tensor input, Tensor weight, Tensor bias, int batch, int seqLen, int inDim, int outDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* wPtr = (float*)weight.DataPointer;
        float* bPtr = (float*)bias.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int inOffset = (b * seqLen + s) * inDim;
                int outOffset = (b * seqLen + s) * outDim;
                for (int o = 0; o < outDim; o++)
                {
                    float sum = bPtr[o];
                    int wOffset = o * inDim;
                    for (int i = 0; i < inDim; i++)
                    {
                        sum += inPtr[inOffset + i] * wPtr[wOffset + i];
                    }
                    outPtr[outOffset + o] = sum;
                }
            }
        }
    }

    /// <summary>Releases all tensor references held by this transformer.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            // Tensor ownership is with the weight loader (safetensors mmap).
            // We just null out our references.
            _timestepLinear1Weight = null;
            _timestepLinear1Bias = null;
            _timestepLinear2Weight = null;
            _timestepLinear2Bias = null;
            _textLinear1Weight = null;
            _textLinear1Bias = null;
            _textLinear2Weight = null;
            _textLinear2Bias = null;
            _contextEmbedWeight = null;
            _contextEmbedBias = null;
            _normOutLinearWeight = null;
            _normOutLinearBias = null;
            _projOutWeight = null;
            _projOutBias = null;
        }
        GC.SuppressFinalize(this);
    }
}
