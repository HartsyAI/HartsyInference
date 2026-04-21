using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers.UNetBlocks;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>UNet2DConditionModel for Stable Diffusion. Takes noisy latents + timestep + text embeddings and predicts the noise (or velocity). Matches HuggingFace diffusers UNet2DConditionModel.</summary>
public sealed class UNet
{
    private readonly UNetConfig _config;

    // conv_in: Conv2d(inChannels, modelChannels, 3, padding=1)
    private Tensor? _convInWeight;
    private Tensor? _convInBias;

    // Timestep embedding
    private readonly TimestepEmbedding _timeEmbedding;

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
        int timeDim = config.ModelChannels * 4; // 1280 for SD1.5
        int numBlocks = blockCh.Length;

        // Timestep embedding: sinusoidal(modelChannels) → MLP → timeDim
        _timeEmbedding = new TimestepEmbedding(config.ModelChannels, timeDim);

        // Down blocks
        _downBlocks = new DownBlock[numBlocks];
        for (int i = 0; i < numBlocks; i++)
        {
            int inCh = i == 0 ? config.ModelChannels : blockCh[i - 1];
            int outCh = blockCh[i];
            bool hasAttn = config.DownBlockHasAttention[i];
            bool hasDown = i < numBlocks - 1; // All except last have downsample

            int numHeads = config.NumAttentionHeads[i];
            _downBlocks[i] = new DownBlock(
                inCh, outCh, timeDim, config.LayersPerBlock,
                hasAttn, hasDown, numHeads, config.CrossAttentionDim);
        }

        // Mid block
        int midCh = blockCh[^1];
        _midResNet0 = new UNetResNetBlock(midCh, midCh, timeDim);
        int midNumHeads = config.NumAttentionHeads[^1];
        _midAttention = new CrossAttentionBlock(midCh, midNumHeads, config.CrossAttentionDim);
        _midResNet1 = new UNetResNetBlock(midCh, midCh, timeDim);

        // Up blocks (reversed channel order with skip connections)
        // Matches diffusers: output_channel = reversed[i], input_channel = reversed[min(i+1, len-1)]
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
            // input_channel from diffusers — this determines the last layer's skip channels
            int inputCh = reversedCh[Math.Min(i + 1, numBlocks - 1)];
            bool hasAttn = config.UpBlockHasAttention[i];
            bool hasUp = i < numBlocks - 1;

            // Up blocks have layers_per_block + 1 ResNet layers
            int upNumHeads = config.NumAttentionHeads[numBlocks - 1 - i];
            _upBlocks[i] = new UpBlock(
                inputCh, outCh, prevOutCh, timeDim, config.LayersPerBlock + 1,
                hasAttn, hasUp, upNumHeads, config.CrossAttentionDim);
        }
    }

    /// <summary>Loads all UNet weights from a dictionary of named tensors.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        string p = string.IsNullOrEmpty(prefix) ? "" : $"{prefix}.";

        _convInWeight = weights[$"{p}conv_in.weight"];
        _convInBias = weights[$"{p}conv_in.bias"];

        _timeEmbedding.LoadWeights(weights, $"{p}time_embedding");

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

    /// <summary>Forward pass: noisy latents [B, 4, H, W] + timestep + text embeddings [B, seqLen, crossDim] → noise prediction [B, 4, H, W].</summary>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeddings)
    {
        int batch = (int)latent.Shape[0];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];

        // 1. Timestep embedding
        Span<float> timesteps = stackalloc float[batch];
        timesteps.Fill(timestep);
        Tensor temb = _timeEmbedding.Forward(backend, timesteps, batch);

        // 2. conv_in
        TensorShape convInShape = new TensorShape(batch, _config.ModelChannels, height, width);
        Tensor hidden = new Tensor(convInShape, DType.F32);
        backend.Conv2D(hidden, latent, _convInWeight!, _convInBias, 1, 1, 1, 1);

        // 3. Down blocks — collect all skip connections
        // Save conv_in output as the first skip (matches diffusers)
        List<Tensor> allSkips = new List<Tensor> { hidden.To(hidden.Device) };
        for (int i = 0; i < _downBlocks.Length; i++)
        {
            (Tensor downOut, List<Tensor> skips) = _downBlocks[i].Forward(backend, hidden, temb, textEmbeddings);
            hidden.Dispose();
            hidden = downOut;
            allSkips.AddRange(skips);
        }

        // 4. Mid block
        Tensor midRes0 = _midResNet0.Forward(backend, hidden, temb);
        hidden.Dispose();

        Tensor midAttn = _midAttention.Forward(backend, midRes0, textEmbeddings);
        midRes0.Dispose();

        Tensor midRes1 = _midResNet1.Forward(backend, midAttn, temb);
        midAttn.Dispose();

        hidden = midRes1;

        // 5. Up blocks — consume skip connections in reverse
        for (int i = 0; i < _upBlocks.Length; i++)
        {
            Tensor upOut = _upBlocks[i].Forward(backend, hidden, temb, textEmbeddings, allSkips);
            hidden.Dispose();
            hidden = upOut;
        }

        temb.Dispose();

        // 6. conv_norm_out → SiLU → conv_out
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
