using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>AutoencoderKL VAE encoder. Converts RGB images [B, 3, H, W] in [-1, 1] to scaled latents [B, latentCh, H/8, W/8]. Mirror of <see cref="VaeDecoder"/>; supports SD1.5, SDXL, SD3, and Flux configurations.
/// <para>Encodes deterministically by returning the posterior mean (mu), not a sample of the diagonal Gaussian. Diffusers' default img2img path samples mu + std·noise via a generator; we use the mean only. The downstream denoise loop adds its own noise via <c>scheduler.AddNoise</c>, which dominates the marginal effect of the absent VAE-side noise. This makes the encoder fully deterministic given a fixed input.</para>
/// <para>Downsamplers match diffusers exactly: asymmetric zero-pad (right/bottom by 1 — <c>F.pad((0,1,0,1))</c>)
/// followed by Conv2d(stride=2, padding=0, kernel=3). A symmetric padding=1 conv produces the same output size for
/// even H/W but samples the OPPOSITE pixel parity — each of the three stride-2 stages shifts the sampling grid by
/// one input pixel, compounding to a ~1-latent-pixel phase error vs the reference encoder. That misalignment left
/// latents visibly off-distribution (Flux Kontext reproduced reference textures as maze/speckle artifacts; corr vs
/// the ComfyUI encode was 0.87 where the aligned encoder reaches 0.999+).</para>
/// </summary>
public sealed class VaeEncoder
{
    private readonly VaeConfig _config;

    // conv_in: Conv2d(3 → block_out_channels[0], 3x3, padding=1)
    private Tensor? _convInWeight;
    private Tensor? _convInBias;

    // down_blocks: N x DownEncoderBlock2D, each with layers_per_block ResNets + optional downsample
    private readonly ResNetBlock2D[][] _downBlockResNets;
    private readonly Tensor?[] _downsampleWeights;
    private readonly Tensor?[] _downsampleBiases;

    // mid_block: ResNet → Attention → ResNet
    private readonly ResNetBlock2D _midResNet0;
    private readonly VaeAttention _midAttention;
    private readonly ResNetBlock2D _midResNet1;

    // conv_norm_out + conv_out (block_out_channels[-1] → 2 * latent_channels, 3x3, padding=1)
    private Tensor? _normOutWeight;
    private Tensor? _normOutBias;
    private Tensor? _convOutWeight;
    private Tensor? _convOutBias;

    // quant_conv (optional, SD1.5/SDXL only): 1x1 conv on the 2*latent_channel moments
    private Tensor? _quantConvWeight;
    private Tensor? _quantConvBias;

    /// <summary>The configuration this encoder was built with.</summary>
    public VaeConfig Config => _config;

    /// <summary>Creates a VAE encoder with the specified configuration.</summary>
    public VaeEncoder(VaeConfig config)
    {
        _config = config;

        int[] blockChannels = config.BlockOutChannels;
        int numBlocks = blockChannels.Length;
        int resnetsPerBlock = config.LayersPerBlock;

        _downBlockResNets = new ResNetBlock2D[numBlocks][];
        _downsampleWeights = new Tensor?[numBlocks];
        _downsampleBiases = new Tensor?[numBlocks];

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
        {
            int outCh = blockChannels[blockIdx];
            _downBlockResNets[blockIdx] = new ResNetBlock2D[resnetsPerBlock];

            for (int resIdx = 0; resIdx < resnetsPerBlock; resIdx++)
            {
                // First ResNet of first block takes conv_in's output (block_channels[0]);
                // first ResNet of each subsequent block takes the previous block's output channels;
                // remaining ResNets stay at outCh.
                int inCh;
                if (resIdx == 0)
                {
                    inCh = blockIdx == 0 ? blockChannels[0] : blockChannels[blockIdx - 1];
                }
                else
                {
                    inCh = outCh;
                }
                _downBlockResNets[blockIdx][resIdx] = new ResNetBlock2D(inCh, outCh, config.NormNumGroups, config.NormEps);
            }
        }

        // Mid block at deepest channel count (mirror of decoder mid block).
        int midChannels = blockChannels[^1];
        _midResNet0 = new ResNetBlock2D(midChannels, midChannels, config.NormNumGroups, config.NormEps);
        _midAttention = new VaeAttention(midChannels, config.NormNumGroups, config.NormEps);
        _midResNet1 = new ResNetBlock2D(midChannels, midChannels, config.NormNumGroups, config.NormEps);
    }

    /// <summary>Loads all encoder weights from a dictionary of named tensors. Keys should match diffusers naming convention ("encoder.conv_in.*", "encoder.down_blocks.{i}.resnets.{j}.*", etc.).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _convInWeight = weights["encoder.conv_in.weight"];
        _convInBias = weights["encoder.conv_in.bias"];

        for (int blockIdx = 0; blockIdx < _downBlockResNets.Length; blockIdx++)
        {
            for (int resIdx = 0; resIdx < _downBlockResNets[blockIdx].Length; resIdx++)
            {
                _downBlockResNets[blockIdx][resIdx].LoadWeights(weights, $"encoder.down_blocks.{blockIdx}.resnets.{resIdx}");
            }

            // Downsample exists on every block except the last one.
            if (blockIdx < _downBlockResNets.Length - 1)
            {
                _downsampleWeights[blockIdx] = weights[$"encoder.down_blocks.{blockIdx}.downsamplers.0.conv.weight"];
                _downsampleBiases[blockIdx] = weights[$"encoder.down_blocks.{blockIdx}.downsamplers.0.conv.bias"];
            }
        }

        _midResNet0.LoadWeights(weights, "encoder.mid_block.resnets.0");
        _midAttention.LoadWeights(weights, "encoder.mid_block.attentions.0");
        _midResNet1.LoadWeights(weights, "encoder.mid_block.resnets.1");

        _normOutWeight = weights["encoder.conv_norm_out.weight"];
        _normOutBias = weights["encoder.conv_norm_out.bias"];
        _convOutWeight = weights["encoder.conv_out.weight"];
        _convOutBias = weights["encoder.conv_out.bias"];

        if (_config.UseQuantConv)
        {
            _quantConvWeight = weights["quant_conv.weight"];
            _quantConvBias = weights["quant_conv.bias"];
        }
    }

    /// <summary>Enumerates all weight tensors held by this encoder and its sub-blocks.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convInWeight is not null) yield return _convInWeight;
        if (_convInBias is not null) yield return _convInBias;

        for (int blockIdx = 0; blockIdx < _downBlockResNets.Length; blockIdx++)
        {
            for (int resIdx = 0; resIdx < _downBlockResNets[blockIdx].Length; resIdx++)
            {
                foreach (Tensor w in _downBlockResNets[blockIdx][resIdx].EnumerateWeights()) yield return w;
            }
            if (_downsampleWeights[blockIdx] is not null) yield return _downsampleWeights[blockIdx]!;
            if (_downsampleBiases[blockIdx] is not null) yield return _downsampleBiases[blockIdx]!;
        }

        foreach (Tensor w in _midResNet0.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midAttention.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midResNet1.EnumerateWeights()) yield return w;

        if (_normOutWeight is not null) yield return _normOutWeight;
        if (_normOutBias is not null) yield return _normOutBias;
        if (_convOutWeight is not null) yield return _convOutWeight;
        if (_convOutBias is not null) yield return _convOutBias;
        if (_quantConvWeight is not null) yield return _quantConvWeight;
        if (_quantConvBias is not null) yield return _quantConvBias;
    }

    /// <summary>Encodes an image tensor [B, 3, H, W] (values in [-1, 1]) to a scaled latent [B, latentCh, H/8, W/8]. Returns the posterior mean only (no sampling). The output has scaling and shift applied per <see cref="VaeConfig"/>: <c>latent = (mu - shift) * scaling</c>.</summary>
    public Tensor Encode(IBackend backend, Tensor image)
    {
        DType dtype = image.DType;
        int batch = (int)image.Shape[0];
        int h = (int)image.Shape[2];
        int w = (int)image.Shape[3];

        int firstCh = _config.BlockOutChannels[0];

        // 1. conv_in: [B, 3, H, W] → [B, firstCh, H, W]
        TensorShape convInOutShape = new TensorShape(batch, firstCh, h, w);
        Tensor hidden = new Tensor(convInOutShape, dtype);
        backend.Conv2D(hidden, image, _convInWeight!, _convInBias, 1, 1, 1, 1);

        // 2. Down blocks
        for (int blockIdx = 0; blockIdx < _downBlockResNets.Length; blockIdx++)
        {
            // ResNet layers
            for (int resIdx = 0; resIdx < _downBlockResNets[blockIdx].Length; resIdx++)
            {
                Tensor resOut = _downBlockResNets[blockIdx][resIdx].Forward(backend, hidden);
                hidden.Dispose();
                hidden = resOut;
            }

            // Downsample (all blocks except last): asymmetric zero-pad (0,1,0,1) + Conv2d(stride=2, padding=0,
            // kernel=3) — the diffusers Downsample2D recipe. See the class doc for why symmetric padding is wrong.
            if (blockIdx < _downBlockResNets.Length - 1)
            {
                int curH = (int)hidden.Shape[2];
                int curW = (int)hidden.Shape[3];
                int curCh = (int)hidden.Shape[1];

                Tensor padded = PadRightBottom(backend, hidden, batch, curCh, curH, curW, dtype);
                hidden.Dispose();

                TensorShape downShape = new TensorShape(batch, curCh, curH / 2, curW / 2);
                Tensor downsampled = new Tensor(downShape, dtype);
                backend.Conv2D(downsampled, padded, _downsampleWeights[blockIdx]!, _downsampleBiases[blockIdx], 2, 2, 0, 0);
                padded.Dispose();
                hidden = downsampled;
            }
        }

        // 3. Mid block: ResNet → Attention → ResNet
        Tensor midOut0 = _midResNet0.Forward(backend, hidden);
        hidden.Dispose();

        Tensor attnOut = _midAttention.Forward(backend, midOut0);
        midOut0.Dispose();

        Tensor midOut1 = _midResNet1.Forward(backend, attnOut);
        attnOut.Dispose();

        hidden = midOut1;

        // 4. Fused GroupNorm+SiLU → conv_out: [B, lastCh, H/8, W/8] → [B, 2*latentCh, H/8, W/8]
        int finalH = (int)hidden.Shape[2];
        int finalW = (int)hidden.Shape[3];
        int lastCh = _config.BlockOutChannels[^1];
        int momentChannels = 2 * _config.LatentChannels;

        TensorShape preOutShape = new TensorShape(batch, lastCh, finalH, finalW);
        Tensor siluOut = new Tensor(preOutShape, dtype);
        backend.GroupNormSilu(siluOut, hidden, _normOutWeight!, _normOutBias!, _config.NormNumGroups, _config.NormEps);
        hidden.Dispose();

        TensorShape momentShape = new TensorShape(batch, momentChannels, finalH, finalW);
        Tensor moments = new Tensor(momentShape, dtype);
        backend.Conv2D(moments, siluOut, _convOutWeight!, _convOutBias, 1, 1, 1, 1);
        siluOut.Dispose();

        // 5. Optional quant_conv (1x1, mixes mu/logvar channels) — SD1.5/SDXL/Flux2 only
        if (_config.UseQuantConv)
        {
            Tensor quantOut = new Tensor(momentShape, dtype);
            backend.Conv2D(quantOut, moments, _quantConvWeight!, _quantConvBias, 1, 1, 0, 0);
            moments.Dispose();
            moments = quantOut;
        }

        // 6. Split [B, 2*latentCh, ...] into mu (first half) and logvar (second half) along channel dim.
        // Discard logvar — we use the mean only.
        TensorShape latentShape = new TensorShape(batch, _config.LatentChannels, finalH, finalW);
        Tensor mu = new Tensor(latentShape, dtype);
        Tensor logvar = new Tensor(latentShape, dtype);
        Tensor[] chunks = [mu, logvar];
        backend.Split(chunks, moments, dim: 1);
        moments.Dispose();
        logvar.Dispose();

        // 7. Apply scaling: latent = (mu - shift) * scaling
        return ApplyScaling(backend, mu);
    }

    /// <summary>Zero-pads a [B, C, H, W] tensor on the right and bottom edges by one (diffusers <c>F.pad((0,1,0,1))</c>) via two device-side concats, keeping the encode free of host glue.</summary>
    internal static Tensor PadRightBottom(IBackend backend, Tensor input, int batch, int channels, int h, int w, DType dtype)
    {
        Tensor colZeros = new Tensor(new TensorShape(batch, channels, h, 1), dtype);
        backend.Fill(colZeros, 0f);
        Tensor widePadded = new Tensor(new TensorShape(batch, channels, h, w + 1), dtype);
        backend.Concat(widePadded, [input, colZeros], dim: 3);
        colZeros.Dispose();

        Tensor rowZeros = new Tensor(new TensorShape(batch, channels, 1, w + 1), dtype);
        backend.Fill(rowZeros, 0f);
        Tensor padded = new Tensor(new TensorShape(batch, channels, h + 1, w + 1), dtype);
        backend.Concat(padded, [widePadded, rowZeros], dim: 2);
        widePadded.Dispose();
        rowZeros.Dispose();
        return padded;
    }

    /// <summary>Applies the inverse of <c>VaeDecoder.UndoScaling</c>: <c>latent = (mu - shift) * scaling</c>. Disposes <paramref name="mu"/> if a new tensor is returned.</summary>
    private Tensor ApplyScaling(IBackend backend, Tensor mu)
    {
        DType dtype = mu.DType;

        Tensor centered;
        if (_config.ShiftFactor.HasValue)
        {
            Tensor shiftTensor = new Tensor(mu.Shape, dtype);
            backend.Fill(shiftTensor, -_config.ShiftFactor.Value);

            centered = new Tensor(mu.Shape, dtype);
            backend.Add(centered, mu, shiftTensor);
            mu.Dispose();
            shiftTensor.Dispose();
        }
        else
        {
            centered = mu;
        }

        Tensor scaled = new Tensor(centered.Shape, dtype);
        backend.Scale(scaled, centered, _config.ScalingFactor);
        centered.Dispose();
        return scaled;
    }
}
