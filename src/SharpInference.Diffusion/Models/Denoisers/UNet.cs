using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers.UNetBlocks;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>UNet2DConditionModel for Stable Diffusion. Supports SD1.5, SDXL base, and SDXL refiner architectures via UNetConfig.</summary>
public sealed class UNet
{
    private readonly UNetConfig _config;

    // conv_in: Conv2d(inChannels, modelChannels, 3, padding=1)
    private Tensor? _convInWeight;
    private Tensor? _convInBias;

    // Timestep embedding
    private readonly TimestepEmbedding _timeEmbedding;

    // ADM conditioning (SDXL only — null for SD1.5)
    private readonly AdditionEmbedding? _addEmbedding;

    // Down blocks
    private readonly DownBlock[] _downBlocks;

    // Mid block: ResNet → CrossAttention → ResNet
    private readonly UNetResNetBlock _midResNet0;
    private readonly CrossAttentionBlock _midAttention;
    private readonly UNetResNetBlock _midResNet1;

    // Up blocks
    private readonly UpBlock[] _upBlocks;

    // conv_norm_out + conv_out
    private Tensor? _normOutWeight;
    private Tensor? _normOutBias;
    private Tensor? _convOutWeight;
    private Tensor? _convOutBias;

    /// <summary>The configuration this UNet was built with.</summary>
    public UNetConfig Config => _config;

    /// <summary>Creates a UNet with the specified configuration.</summary>
    public UNet(UNetConfig config)
    {
        _config = config;

        int[] blockCh = config.BlockOutChannels;
        int timeDim = config.ModelChannels * 4; // 1280 for SD1.5/SDXL base, 1536 for SDXL refiner
        int numBlocks = blockCh.Length;

        // Timestep embedding: sinusoidal(modelChannels) → MLP → timeDim
        _timeEmbedding = new TimestepEmbedding(config.ModelChannels, timeDim);

        // ADM conditioning for SDXL (micro-conditioning: size/crop/target scalars + pooled text)
        if (config.AdmInChannels > 0)
        {
            _addEmbedding = new AdditionEmbedding(config.AdmInChannels, timeDim, config.AdditionTimeEmbedDim);
        }

        // Down blocks
        _downBlocks = new DownBlock[numBlocks];
        for (int i = 0; i < numBlocks; i++)
        {
            int inCh = i == 0 ? config.ModelChannels : blockCh[i - 1];
            int outCh = blockCh[i];
            bool hasAttn = config.DownBlockHasAttention[i];
            bool hasDown = i < numBlocks - 1;

            int numHeads = config.NumAttentionHeads[i];
            int transformerLayers = config.TransformerLayersPerBlock[i];
            _downBlocks[i] = new DownBlock(
                inCh, outCh, timeDim, config.LayersPerBlock,
                hasAttn, hasDown, numHeads, config.CrossAttentionDim, transformerLayers);
        }

        // Mid block — uses last level's config
        int midCh = blockCh[^1];
        _midResNet0 = new UNetResNetBlock(midCh, midCh, timeDim);
        int midNumHeads = config.NumAttentionHeads[^1];
        int midTransformerLayers = config.TransformerLayersPerBlock[^1];
        _midAttention = new CrossAttentionBlock(midCh, midNumHeads, config.CrossAttentionDim, midTransformerLayers);
        _midResNet1 = new UNetResNetBlock(midCh, midCh, timeDim);

        // Up blocks (reversed channel order with skip connections)
        _upBlocks = new UpBlock[numBlocks];
        int[] reversedCh = new int[numBlocks];
        for (int i = 0; i < numBlocks; i++)
        {
            reversedCh[i] = blockCh[numBlocks - 1 - i];
        }

        for (int i = 0; i < numBlocks; i++)
        {
            int outCh = reversedCh[i];
            int prevOutCh = i == 0 ? midCh : reversedCh[i - 1];
            int inputCh = reversedCh[Math.Min(i + 1, numBlocks - 1)];
            bool hasAttn = config.UpBlockHasAttention[i];
            bool hasUp = i < numBlocks - 1;

            int upLevelIdx = numBlocks - 1 - i;
            int upNumHeads = config.NumAttentionHeads[upLevelIdx];
            int upTransformerLayers = config.TransformerLayersPerBlock[upLevelIdx];
            _upBlocks[i] = new UpBlock(
                inputCh, outCh, prevOutCh, timeDim, config.LayersPerBlock + 1,
                hasAttn, hasUp, upNumHeads, config.CrossAttentionDim, upTransformerLayers);
        }
    }

    /// <summary>Loads all UNet weights from a dictionary of named tensors.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        string p = string.IsNullOrEmpty(prefix) ? "" : $"{prefix}.";

        _convInWeight = weights[$"{p}conv_in.weight"];
        _convInBias = weights[$"{p}conv_in.bias"];

        _timeEmbedding.LoadWeights(weights, $"{p}time_embedding");

        if (_addEmbedding is not null)
        {
            _addEmbedding.LoadWeights(weights, $"{p}add_embedding");
        }

        for (int i = 0; i < _downBlocks.Length; i++)
        {
            _downBlocks[i].LoadWeights(weights, $"{p}down_blocks.{i}");
        }

        _midResNet0.LoadWeights(weights, $"{p}mid_block.resnets.0");
        _midAttention.LoadWeights(weights, $"{p}mid_block.attentions.0");
        _midResNet1.LoadWeights(weights, $"{p}mid_block.resnets.1");

        for (int i = 0; i < _upBlocks.Length; i++)
        {
            _upBlocks[i].LoadWeights(weights, $"{p}up_blocks.{i}");
        }

        _normOutWeight = weights[$"{p}conv_norm_out.weight"];
        _normOutBias = weights[$"{p}conv_norm_out.bias"];
        _convOutWeight = weights[$"{p}conv_out.weight"];
        _convOutBias = weights[$"{p}conv_out.bias"];
    }

    /// <summary>Enumerates all weight tensors held by this UNet and its sub-blocks.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convInWeight is not null) yield return _convInWeight;
        if (_convInBias is not null) yield return _convInBias;
        foreach (Tensor w in _timeEmbedding.EnumerateWeights()) yield return w;
        if (_addEmbedding is not null)
        {
            foreach (Tensor w in _addEmbedding.EnumerateWeights()) yield return w;
        }
        for (int i = 0; i < _downBlocks.Length; i++)
        {
            foreach (Tensor w in _downBlocks[i].EnumerateWeights()) yield return w;
        }
        foreach (Tensor w in _midResNet0.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midAttention.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midResNet1.EnumerateWeights()) yield return w;
        for (int i = 0; i < _upBlocks.Length; i++)
        {
            foreach (Tensor w in _upBlocks[i].EnumerateWeights()) yield return w;
        }
        if (_normOutWeight is not null) yield return _normOutWeight;
        if (_normOutBias is not null) yield return _normOutBias;
        if (_convOutWeight is not null) yield return _convOutWeight;
        if (_convOutBias is not null) yield return _convOutBias;
    }

    /// <summary>Forward pass for SD1.5 (no ADM conditioning). Noisy latents [B, 4, H, W] + timestep + text embeddings [B, seqLen, crossDim] → noise prediction [B, 4, H, W].</summary>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeddings)
    {
        return Forward(backend, latent, timestep, textEmbeddings, null, default);
    }

    /// <summary>Forward pass with optional SDXL ADM conditioning. For SD1.5, pooledTextEmb and sizeCondition can be null/empty.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="latent">Noisy latent [B, 4, H, W].</param>
    /// <param name="timestep">Current denoising timestep (scalar).</param>
    /// <param name="textEmbeddings">Text encoder hidden states [B, seqLen, crossAttentionDim].</param>
    /// <param name="pooledTextEmb">SDXL: pooled text embedding from CLIP-G [B, 1280]. Null for SD1.5.</param>
    /// <param name="sizeCondition">SDXL: micro-conditioning scalars [origH, origW, cropTop, cropLeft, targetH, targetW]. Empty for SD1.5.</param>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeddings, Tensor? pooledTextEmb, ReadOnlySpan<float> sizeCondition)
    {
        int batch = (int)latent.Shape[0];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];

        // 1. Timestep embedding
        Span<float> timesteps = stackalloc float[batch];
        timesteps.Fill(timestep);
        Tensor temb = _timeEmbedding.Forward(backend, timesteps, batch);

        // 2. ADM conditioning (SDXL only): add micro-conditioning to timestep embedding
        if (_addEmbedding is not null && pooledTextEmb is not null)
        {
            Tensor addEmb = _addEmbedding.Forward(backend, pooledTextEmb, sizeCondition, batch);
            Tensor combinedTemb = new Tensor(temb.Shape, DType.F32);
            backend.Add(combinedTemb, temb, addEmb);
            temb.Dispose();
            addEmb.Dispose();
            temb = combinedTemb;
        }

        // 3. conv_in
        TensorShape convInShape = new TensorShape(batch, _config.ModelChannels, height, width);
        Tensor hidden = new Tensor(convInShape, DType.F32);
        backend.Conv2D(hidden, latent, _convInWeight!, _convInBias, 1, 1, 1, 1);

        // 4. Down blocks — collect all skip connections
        List<Tensor> allSkips = new List<Tensor> { hidden.To(hidden.Device) };
        for (int i = 0; i < _downBlocks.Length; i++)
        {
            (Tensor downOut, List<Tensor> skips) = _downBlocks[i].Forward(backend, hidden, temb, textEmbeddings);
            hidden.Dispose();
            hidden = downOut;
            allSkips.AddRange(skips);
        }

        // 5. Mid block
        Tensor midRes0 = _midResNet0.Forward(backend, hidden, temb);
        hidden.Dispose();

        Tensor midAttn = _midAttention.Forward(backend, midRes0, textEmbeddings);
        midRes0.Dispose();

        Tensor midRes1 = _midResNet1.Forward(backend, midAttn, temb);
        midAttn.Dispose();

        hidden = midRes1;

        // 6. Up blocks — consume skip connections in reverse
        for (int i = 0; i < _upBlocks.Length; i++)
        {
            Tensor upOut = _upBlocks[i].Forward(backend, hidden, temb, textEmbeddings, allSkips);
            hidden.Dispose();
            hidden = upOut;
        }

        temb.Dispose();

        // 7. conv_norm_out → SiLU → conv_out
        int finalH = (int)hidden.Shape[2];
        int finalW = (int)hidden.Shape[3];
        int finalCh = _config.ModelChannels;

        TensorShape finalShape = new TensorShape(batch, finalCh, finalH, finalW);
        Tensor normOut = new Tensor(finalShape, DType.F32);
        backend.GroupNorm(normOut, hidden, _normOutWeight!, _normOutBias!, _config.NormNumGroups, _config.NormEps);
        hidden.Dispose();

        Tensor siluOut = new Tensor(finalShape, DType.F32);
        backend.Silu(siluOut, normOut);
        normOut.Dispose();

        TensorShape outShape = new TensorShape(batch, _config.OutChannels, finalH, finalW);
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Conv2D(output, siluOut, _convOutWeight!, _convOutBias, 1, 1, 1, 1);
        siluOut.Dispose();

        return output;
    }
}
