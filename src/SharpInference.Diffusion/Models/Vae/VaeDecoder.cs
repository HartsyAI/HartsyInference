using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>AutoencoderKL VAE decoder. Converts latent representations [B, C, H/8, W/8] to RGB images [B, 3, H, W]. Supports SD1.5, SDXL, SD3, and Flux configurations.</summary>
public sealed class VaeDecoder
{
    private readonly VaeConfig _config;

    // post_quant_conv (optional, SD1.5/SDXL only)
    private Tensor? _postQuantConvWeight;
    private Tensor? _postQuantConvBias;

    // conv_in: Conv2d(latent_channels → block_out_channels[-1], 3x3, padding=1)
    private Tensor? _convInWeight;
    private Tensor? _convInBias;

    // mid_block: ResNet → Attention → ResNet
    private readonly ResNetBlock2D _midResNet0;
    private readonly VaeAttention _midAttention;
    private readonly ResNetBlock2D _midResNet1;

    // up_blocks: 4x UpDecoderBlock2D, each with (layers_per_block + 1) ResNets + optional upsample
    private readonly ResNetBlock2D[][] _upBlockResNets;
    private readonly Tensor?[] _upsampleWeights;
    private readonly Tensor?[] _upsampleBiases;

    // conv_norm_out + conv_out
    private Tensor? _normOutWeight;
    private Tensor? _normOutBias;
    private Tensor? _convOutWeight;
    private Tensor? _convOutBias;

    /// <summary>The configuration this decoder was built with.</summary>
    public VaeConfig Config => _config;

    /// <summary>Creates a VAE decoder with the specified configuration.</summary>
    public VaeDecoder(VaeConfig config)
    {
        _config = config;

        int[] blockChannels = config.BlockOutChannels;
        int midChannels = blockChannels[^1];

        // Mid-block: two ResNet blocks with attention between them (all at midChannels)
        _midResNet0 = new ResNetBlock2D(midChannels, midChannels, config.NormNumGroups, config.NormEps);
        _midAttention = new VaeAttention(midChannels, config.NormNumGroups, config.NormEps);
        _midResNet1 = new ResNetBlock2D(midChannels, midChannels, config.NormNumGroups, config.NormEps);

        // Up-blocks: reversed channel order [512, 512, 256, 128]
        int numBlocks = blockChannels.Length;
        int resnetsPerBlock = config.LayersPerBlock + 1;
        _upBlockResNets = new ResNetBlock2D[numBlocks][];
        _upsampleWeights = new Tensor?[numBlocks];
        _upsampleBiases = new Tensor?[numBlocks];

        // Up-blocks process in reversed channel order
        int[] reversedChannels = new int[numBlocks];
        for (int i = 0; i < numBlocks; i++)
        {
            reversedChannels[i] = blockChannels[numBlocks - 1 - i];
        }

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
        {
            int outCh = reversedChannels[blockIdx];
            _upBlockResNets[blockIdx] = new ResNetBlock2D[resnetsPerBlock];

            for (int resIdx = 0; resIdx < resnetsPerBlock; resIdx++)
            {
                // First ResNet in first block takes midChannels as input
                // First ResNet in subsequent blocks takes previous block's output channels
                int inCh;
                if (resIdx == 0)
                {
                    inCh = blockIdx == 0 ? midChannels : reversedChannels[blockIdx - 1];
                }
                else
                {
                    inCh = outCh;
                }
                _upBlockResNets[blockIdx][resIdx] = new ResNetBlock2D(inCh, outCh, config.NormNumGroups, config.NormEps);
            }
        }
    }

    /// <summary>Loads all decoder weights from a dictionary of named tensors. Keys should match diffusers naming convention.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        // post_quant_conv (optional)
        if (_config.UsePostQuantConv)
        {
            _postQuantConvWeight = weights["post_quant_conv.weight"];
            _postQuantConvBias = weights["post_quant_conv.bias"];
        }

        // conv_in
        _convInWeight = weights["decoder.conv_in.weight"];
        _convInBias = weights["decoder.conv_in.bias"];

        // mid_block
        _midResNet0.LoadWeights(weights, "decoder.mid_block.resnets.0");
        _midAttention.LoadWeights(weights, "decoder.mid_block.attentions.0");
        _midResNet1.LoadWeights(weights, "decoder.mid_block.resnets.1");

        // up_blocks
        for (int blockIdx = 0; blockIdx < _upBlockResNets.Length; blockIdx++)
        {
            for (int resIdx = 0; resIdx < _upBlockResNets[blockIdx].Length; resIdx++)
            {
                _upBlockResNets[blockIdx][resIdx].LoadWeights(weights, $"decoder.up_blocks.{blockIdx}.resnets.{resIdx}");
            }

            // Upsample exists on all blocks except the last one
            if (blockIdx < _upBlockResNets.Length - 1)
            {
                _upsampleWeights[blockIdx] = weights[$"decoder.up_blocks.{blockIdx}.upsamplers.0.conv.weight"];
                _upsampleBiases[blockIdx] = weights[$"decoder.up_blocks.{blockIdx}.upsamplers.0.conv.bias"];
            }
        }

        // conv_norm_out + conv_out
        _normOutWeight = weights["decoder.conv_norm_out.weight"];
        _normOutBias = weights["decoder.conv_norm_out.bias"];
        _convOutWeight = weights["decoder.conv_out.weight"];
        _convOutBias = weights["decoder.conv_out.bias"];
    }

    /// <summary>Enumerates all weight tensors held by this decoder and its sub-blocks.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_postQuantConvWeight is not null) yield return _postQuantConvWeight;
        if (_postQuantConvBias is not null) yield return _postQuantConvBias;
        if (_convInWeight is not null) yield return _convInWeight;
        if (_convInBias is not null) yield return _convInBias;
        foreach (Tensor w in _midResNet0.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midAttention.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midResNet1.EnumerateWeights()) yield return w;
        for (int blockIdx = 0; blockIdx < _upBlockResNets.Length; blockIdx++)
        {
            for (int resIdx = 0; resIdx < _upBlockResNets[blockIdx].Length; resIdx++)
            {
                foreach (Tensor w in _upBlockResNets[blockIdx][resIdx].EnumerateWeights()) yield return w;
            }
            if (_upsampleWeights[blockIdx] is not null) yield return _upsampleWeights[blockIdx]!;
            if (_upsampleBiases[blockIdx] is not null) yield return _upsampleBiases[blockIdx]!;
        }
        if (_normOutWeight is not null) yield return _normOutWeight;
        if (_normOutBias is not null) yield return _normOutBias;
        if (_convOutWeight is not null) yield return _convOutWeight;
        if (_convOutBias is not null) yield return _convOutBias;
    }

    /// <summary>Decodes latent tensor [B, latentCh, H, W] to RGB image [B, 3, H*8, W*8]. Applies inverse scaling before decoding.</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        // 1. Undo scaling: z = latent / scaling_factor + shift_factor
        Tensor z = UndoScaling(backend, latent);

        // 2. Optional post_quant_conv (1x1 conv)
        if (_config.UsePostQuantConv)
        {
            Tensor quantOut = new Tensor(z.Shape, DType.F32);
            backend.Conv2D(quantOut, z, _postQuantConvWeight!, _postQuantConvBias, 1, 1, 0, 0);
            z.Dispose();
            z = quantOut;
        }

        // 3. conv_in: [B, latentCh, H, W] → [B, 512, H, W]
        int batch = (int)z.Shape[0];
        int h = (int)z.Shape[2];
        int w = (int)z.Shape[3];
        int midCh = _config.BlockOutChannels[^1];

        TensorShape convInOutShape = new TensorShape(batch, midCh, h, w);
        Tensor hidden = new Tensor(convInOutShape, DType.F32);
        backend.Conv2D(hidden, z, _convInWeight!, _convInBias, 1, 1, 1, 1);
        z.Dispose();

        // 4. Mid-block: ResNet → Attention → ResNet
        Tensor midOut0 = _midResNet0.Forward(backend, hidden);
        hidden.Dispose();

        Tensor attnOut = _midAttention.Forward(backend, midOut0);
        midOut0.Dispose();

        Tensor midOut1 = _midResNet1.Forward(backend, attnOut);
        attnOut.Dispose();

        hidden = midOut1;

        // 5. Up-blocks
        int[] reversedChannels = new int[_config.BlockOutChannels.Length];
        for (int i = 0; i < reversedChannels.Length; i++)
        {
            reversedChannels[i] = _config.BlockOutChannels[^(i + 1)];
        }

        for (int blockIdx = 0; blockIdx < _upBlockResNets.Length; blockIdx++)
        {
            // ResNet layers
            for (int resIdx = 0; resIdx < _upBlockResNets[blockIdx].Length; resIdx++)
            {
                Tensor resOut = _upBlockResNets[blockIdx][resIdx].Forward(backend, hidden);
                hidden.Dispose();
                hidden = resOut;
            }

            // Upsample (all blocks except last): nearest-neighbor 2x → Conv2d(3x3)
            if (blockIdx < _upBlockResNets.Length - 1)
            {
                int curH = (int)hidden.Shape[2];
                int curW = (int)hidden.Shape[3];
                int curCh = (int)hidden.Shape[1];

                TensorShape upShape = new TensorShape(batch, curCh, curH * 2, curW * 2);
                Tensor upsampled = new Tensor(upShape, DType.F32);
                backend.UpsampleNearest2D(upsampled, hidden, 2, 2);
                hidden.Dispose();

                Tensor convUp = new Tensor(upShape, DType.F32);
                backend.Conv2D(convUp, upsampled, _upsampleWeights[blockIdx]!, _upsampleBiases[blockIdx], 1, 1, 1, 1);
                upsampled.Dispose();

                hidden = convUp;
            }
        }

        // 6. conv_norm_out → SiLU → conv_out
        int finalH = (int)hidden.Shape[2];
        int finalW = (int)hidden.Shape[3];
        int finalCh = _config.BlockOutChannels[0];

        TensorShape finalShape = new TensorShape(batch, finalCh, finalH, finalW);
        Tensor normOut = new Tensor(finalShape, DType.F32);
        backend.GroupNorm(normOut, hidden, _normOutWeight!, _normOutBias!, _config.NormNumGroups, _config.NormEps);
        hidden.Dispose();

        Tensor siluOut = new Tensor(finalShape, DType.F32);
        backend.Silu(siluOut, normOut);
        normOut.Dispose();

        TensorShape rgbShape = new TensorShape(batch, 3, finalH, finalW);
        Tensor output = new Tensor(rgbShape, DType.F32);
        backend.Conv2D(output, siluOut, _convOutWeight!, _convOutBias, 1, 1, 1, 1);
        siluOut.Dispose();

        return output;
    }

    /// <summary>Undoes the latent scaling: z = latent / scaling_factor + shift_factor.</summary>
    private Tensor UndoScaling(IBackend backend, Tensor latent)
    {
        float invScale = 1.0f / _config.ScalingFactor;
        Tensor scaled = new Tensor(latent.Shape, DType.F32);
        backend.Scale(scaled, latent, invScale);

        if (_config.ShiftFactor.HasValue)
        {
            // Add shift_factor to every element
            Tensor shiftTensor = new Tensor(latent.Shape, DType.F32);
            backend.Fill(shiftTensor, _config.ShiftFactor.Value);

            Tensor shifted = new Tensor(latent.Shape, DType.F32);
            backend.Add(shifted, scaled, shiftTensor);
            scaled.Dispose();
            shiftTensor.Dispose();

            return shifted;
        }

        return scaled;
    }
}
