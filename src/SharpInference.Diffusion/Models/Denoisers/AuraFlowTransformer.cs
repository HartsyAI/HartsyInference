using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>AuraFlow MMDiT transformer. Processes image patches and T5 text tokens through joint double-stream blocks followed by single-stream image-only blocks. Uses SDXL-compatible VAE (4-channel latent, patch size 2).</summary>
public sealed unsafe class AuraFlowTransformer : IDisposable
{
    private readonly AuraFlowConfig _config;
    private readonly PatchEmbed _patchEmbed;
    private readonly AuraFlowJointBlock[] _doubleBlocks;
    private readonly AuraFlowSingleBlock[] _singleBlocks;
    private int _disposed;

    // Timestep embedding: sinusoidal → Linear → SiLU → Linear
    private Tensor? _timestepLinear1Weight, _timestepLinear1Bias;
    private Tensor? _timestepLinear2Weight, _timestepLinear2Bias;

    // Context embedder: Linear(context_dim, hidden_size)
    private Tensor? _contextEmbedWeight, _contextEmbedBias;

    // Positional embedding (learned): [1, max_seq, hidden_size]
    private Tensor? _posEmbed;

    // Register tokens: [1, num_register, hidden_size]
    private Tensor? _registerTokens;

    // Final layer: AdaLN-Continuous + Linear projection
    private Tensor? _normOutLinearWeight, _normOutLinearBias;
    private Tensor? _projOutWeight, _projOutBias;

    /// <summary>Creates an AuraFlow transformer from configuration.</summary>
    public AuraFlowTransformer(AuraFlowConfig config)
    {
        _config = config;

        _patchEmbed = new PatchEmbed(config.PatchSize, config.InChannels, config.HiddenSize);

        int ffDim = config.HiddenSize * 4;

        _doubleBlocks = new AuraFlowJointBlock[config.NumDoubleBlocks];
        for (int i = 0; i < config.NumDoubleBlocks; i++)
        {
            _doubleBlocks[i] = new AuraFlowJointBlock(
                config.HiddenSize, config.NumHeads, ffDim,
                config.UseQkNorm, config.QkNormEps);
        }

        _singleBlocks = new AuraFlowSingleBlock[config.NumSingleBlocks];
        for (int i = 0; i < config.NumSingleBlocks; i++)
        {
            _singleBlocks[i] = new AuraFlowSingleBlock(
                config.HiddenSize, config.NumHeads, ffDim,
                config.UseQkNorm, config.QkNormEps);
        }
    }

    /// <summary>Loads all transformer weights from named tensors using diffusers naming.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _patchEmbed.LoadWeights(
            weights["pos_embed.proj.weight"],
            weights["pos_embed.proj.bias"],
            weights.ContainsKey("pos_embed.pos_embed") ? weights["pos_embed.pos_embed"] : null);

        _timestepLinear1Weight = weights["time_step_proj.linear_1.weight"];
        _timestepLinear1Bias = weights["time_step_proj.linear_1.bias"];
        _timestepLinear2Weight = weights["time_step_proj.linear_2.weight"];
        _timestepLinear2Bias = weights["time_step_proj.linear_2.bias"];

        _contextEmbedWeight = weights["context_embedder.weight"];
        _contextEmbedBias = weights["context_embedder.bias"];

        if (weights.ContainsKey("register_tokens"))
            _registerTokens = weights["register_tokens"];

        for (int i = 0; i < _config.NumDoubleBlocks; i++)
            _doubleBlocks[i].LoadWeights(weights, $"joint_transformer_blocks.{i}");

        for (int i = 0; i < _config.NumSingleBlocks; i++)
            _singleBlocks[i].LoadWeights(weights, $"single_transformer_blocks.{i}");

        _normOutLinearWeight = weights["norm_out.linear.weight"];
        _normOutLinearBias = weights["norm_out.linear.bias"];
        _projOutWeight = weights["proj_out.weight"];
        _projOutBias = weights["proj_out.bias"];
    }

    /// <summary>Forward pass: predicts noise/velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="latent">Noisy latent [B, 4, H, W].</param>
    /// <param name="timestep">Timestep value.</param>
    /// <param name="context">T5 text embeddings [B, seqLen, 4096].</param>
    /// <returns>Predicted noise/velocity [B, 4, H, W].</returns>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor context)
    {
        int batch = (int)latent.Shape[0];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];

        // ── 1. Patch embed latent → image tokens ────────────────────────
        Tensor imageTokens = _patchEmbed.Forward(backend, latent);
        int imgSeqLen = (int)imageTokens.Shape[1];
        (int gridH, int gridW) = _patchEmbed.GetGridSize(height, width);

        // ── 2. Compute timestep embedding ────────────────────────────────
        Tensor temb = ComputeTimestepEmbedding(backend, timestep, batch);

        // ── 3. Project context: [B, S, 4096] → [B, S, hidden] ──────────
        // TODO: Project context through context_embedder linear layer using backend.Linear

        // ── 4. Joint double-stream blocks ────────────────────────────────
        Tensor currentImage = imageTokens;
        Tensor currentText = context;

        for (int i = 0; i < _config.NumDoubleBlocks; i++)
        {
            // TODO: Run AuraFlowJointBlock.Forward once implemented
            // (Tensor newImage, Tensor newText) = _doubleBlocks[i].Forward(backend, currentImage, currentText, temb);
            // Dispose old tensors, update currentImage/currentText
        }

        // ── 5. Single-stream blocks (image-only) ────────────────────────
        for (int i = 0; i < _config.NumSingleBlocks; i++)
        {
            // TODO: Run AuraFlowSingleBlock.Forward once implemented
            // Tensor newImage = _singleBlocks[i].Forward(backend, currentImage, temb);
            // Dispose old, update currentImage
        }

        // ── 6. Final layer: AdaLN-Continuous + projection ────────────────
        // TODO: Apply final AdaLN + linear projection → [B, seqLen, patchDim]

        // ── 7. Unpatchify → [B, 4, H, W] ────────────────────────────────
        // TODO: Use Unpatchify to reconstruct spatial tensor

        temb.Dispose();
        throw new NotImplementedException("AuraFlowTransformer.Forward requires block forward implementations");
    }

    private Tensor ComputeTimestepEmbedding(IBackend backend, float timestep, int batch)
    {
        int hidden = _config.HiddenSize;

        TensorShape sinShape = new TensorShape(batch, 256);
        Tensor sinEmbed = new Tensor(sinShape, DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinEmbed, timestep, batch);

        TensorShape hidShape = new TensorShape(batch, hidden);
        Tensor t1 = new Tensor(hidShape, DType.F32);
        DiTUtils.LinearProject1D(t1, sinEmbed, _timestepLinear1Weight!, _timestepLinear1Bias!, batch, 256, hidden);
        sinEmbed.Dispose();

        Tensor t1Activated = new Tensor(hidShape, DType.F32);
        backend.Silu(t1Activated, t1);
        t1.Dispose();

        Tensor tEmb = new Tensor(hidShape, DType.F32);
        DiTUtils.LinearProject1D(tEmb, t1Activated, _timestepLinear2Weight!, _timestepLinear2Bias!, batch, hidden, hidden);
        t1Activated.Dispose();

        return tEmb;
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        // TODO: PatchEmbed weights (proj.weight, proj.bias) are stored internally — add PatchEmbed.EnumerateWeights()
        if (_contextEmbedWeight is not null) yield return _contextEmbedWeight;
        if (_contextEmbedBias is not null) yield return _contextEmbedBias;
        if (_timestepLinear1Weight is not null) yield return _timestepLinear1Weight;
        if (_timestepLinear1Bias is not null) yield return _timestepLinear1Bias;
        if (_timestepLinear2Weight is not null) yield return _timestepLinear2Weight;
        if (_timestepLinear2Bias is not null) yield return _timestepLinear2Bias;
        if (_posEmbed is not null) yield return _posEmbed;
        if (_registerTokens is not null) yield return _registerTokens;
        if (_normOutLinearWeight is not null) yield return _normOutLinearWeight;
        if (_normOutLinearBias is not null) yield return _normOutLinearBias;
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;

        foreach (AuraFlowJointBlock block in _doubleBlocks)
            foreach (Tensor t in block.EnumerateWeights()) yield return t;
        foreach (AuraFlowSingleBlock block in _singleBlocks)
            foreach (Tensor t in block.EnumerateWeights()) yield return t;
    }

    /// <summary>Releases all tensor references held by this transformer.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _timestepLinear1Weight = null;
            _timestepLinear1Bias = null;
            _timestepLinear2Weight = null;
            _timestepLinear2Bias = null;
            _contextEmbedWeight = null;
            _contextEmbedBias = null;
            _posEmbed = null;
            _registerTokens = null;
            _normOutLinearWeight = null;
            _normOutLinearBias = null;
            _projOutWeight = null;
            _projOutBias = null;
        }
        GC.SuppressFinalize(this);
    }

}
