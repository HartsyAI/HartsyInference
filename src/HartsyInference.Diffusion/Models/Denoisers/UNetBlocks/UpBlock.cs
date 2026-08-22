using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;

/// <summary>UNet up block: sequence of (concat skip + ResNet + optional CrossAttention) layers followed by an optional upsample.</summary>
public sealed class UpBlock
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _prevOutChannels;
    private readonly int _numLayers;
    private readonly bool _hasAttention;
    private readonly bool _hasUpsample;

    private readonly UNetResNetBlock[] _resnets;
    private readonly CrossAttentionBlock?[] _attentions;

    /// <summary>Total cross-attention sub-layer count this block exposes for IP-Adapter purposes — sum of <see cref="CrossAttentionBlock.NumTransformerBlocks"/> across all attention layers in the block. <c>0</c> when <c>hasAttention=false</c>; the UNet uses this to advance the IPA layer-index cursor between blocks.</summary>
    public int CrossAttentionLayerCount
    {
        get
        {
            if (!_hasAttention) return 0;
            int count = 0;
            for (int i = 0; i < _numLayers; i++)
            {
                if (_attentions[i] is not null) count += _attentions[i]!.NumTransformerBlocks;
            }
            return count;
        }
    }

    // Upsample: nearest-neighbor 2x → Conv2d(outCh, outCh, 3, padding=1)
    private Tensor? _upsampleWeight;
    private Tensor? _upsampleBias;

    /// <summary>Creates a UNet up block.</summary>
    /// <param name="inChannels">Input channels from previous up block (or mid block).</param>
    /// <param name="outChannels">Output channels for this block.</param>
    /// <param name="prevOutChannels">Channels from the previous layer (for computing ResNet input with skip concatenation).</param>
    /// <param name="timeDim">Timestep embedding dimension.</param>
    /// <param name="numLayers">Number of ResNet (+attention) layers (typically layers_per_block + 1 = 3 for SD1.5).</param>
    /// <param name="hasAttention">Whether this block has cross-attention.</param>
    /// <param name="hasUpsample">Whether this block has an upsample layer.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="crossAttentionDim">Cross-attention context dimension.</param>
    public UpBlock(int inChannels, int outChannels, int prevOutChannels, int timeDim, int numLayers, bool hasAttention, bool hasUpsample, int numHeads = 8, int crossAttentionDim = 768, int numTransformerBlocks = 1)
    {
        _inChannels = inChannels;
        _outChannels = outChannels;
        _prevOutChannels = prevOutChannels;
        _numLayers = numLayers;
        _hasAttention = hasAttention;
        _hasUpsample = hasUpsample;

        _resnets = new UNetResNetBlock[numLayers];
        _attentions = new CrossAttentionBlock?[numLayers];

        for (int i = 0; i < numLayers; i++)
        {
            // Matches diffusers: skip channels vary per layer
            // Last layer uses inChannels (from deeper block), others use outChannels
            int resSkipCh = (i == numLayers - 1) ? inChannels : outChannels;
            int resMainCh = (i == 0) ? prevOutChannels : outChannels;
            int resInCh = resMainCh + resSkipCh;

            _resnets[i] = new UNetResNetBlock(resInCh, outChannels, timeDim);

            if (hasAttention)
            {
                _attentions[i] = new CrossAttentionBlock(outChannels, numHeads, crossAttentionDim, numTransformerBlocks);
            }
        }
    }

    /// <summary>Loads weights from named tensors.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        for (int i = 0; i < _numLayers; i++)
        {
            _resnets[i].LoadWeights(weights, $"{prefix}.resnets.{i}");

            if (_hasAttention)
            {
                _attentions[i]!.LoadWeights(weights, $"{prefix}.attentions.{i}");
            }
        }

        if (_hasUpsample)
        {
            _upsampleWeight = weights[$"{prefix}.upsamplers.0.conv.weight"];
            _upsampleBias = weights[$"{prefix}.upsamplers.0.conv.bias"];
        }
    }

    /// <summary>Enumerates all weight tensors held by this block and its sub-blocks.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            foreach (Tensor w in _resnets[i].EnumerateWeights()) yield return w;
            if (_attentions[i] is not null)
            {
                foreach (Tensor w in _attentions[i]!.EnumerateWeights()) yield return w;
            }
        }
        if (_upsampleWeight is not null) yield return _upsampleWeight;
        if (_upsampleBias is not null) yield return _upsampleBias;
    }

    /// <summary>Forward pass with skip connections from the corresponding down block. No IPA injection.</summary>
    public Tensor Forward(IBackend backend, Tensor input, Tensor temb, Tensor context, List<Tensor> skips)
    {
        return Forward(backend, input, temb, context, skips, ipaImageTokens: null, ipaToKIpAll: null, ipaToVIpAll: null, ipaStartIndex: 0, ipaScalePerLayer: null);
    }

    /// <summary>Forward pass with skip connections + optional IP-Adapter image-attention injection. Block consumes <see cref="CrossAttentionLayerCount"/> entries from the K/V lists starting at <paramref name="ipaStartIndex"/>; per-layer scale comes from <paramref name="ipaScalePerLayer"/> at the matching flat indices.</summary>
    public Tensor Forward(IBackend backend, Tensor input, Tensor temb, Tensor context, List<Tensor> skips,
        Tensor? ipaImageTokens, IReadOnlyList<Tensor>? ipaToKIpAll, IReadOnlyList<Tensor>? ipaToVIpAll, int ipaStartIndex, IReadOnlyList<float>? ipaScalePerLayer)
    {
        Tensor hidden = input;
        int ipaOffset = ipaStartIndex;

        for (int i = 0; i < _numLayers; i++)
        {
            // Pop skip connection (last element = most recent from down block)
            Tensor skip = skips[^1];
            skips.RemoveAt(skips.Count - 1);

            Tensor concatenated = ConcatChannels(backend, hidden, skip);
            if (hidden != input) hidden.Dispose();
            skip.Dispose();

            Tensor resOut = _resnets[i].Forward(backend, concatenated, temb);
            concatenated.Dispose();
            hidden = resOut;

            if (_hasAttention && _attentions[i] is not null)
            {
                Tensor attnOut = _attentions[i]!.Forward(backend, hidden, context,
                    ipaImageTokens, ipaToKIpAll, ipaToVIpAll, ipaOffset, ipaScalePerLayer);
                ipaOffset += _attentions[i]!.NumTransformerBlocks;
                hidden.Dispose();
                hidden = attnOut;
            }
        }

        if (_hasUpsample)
        {
            int batch = (int)hidden.Shape[0];
            int ch = (int)hidden.Shape[1];
            int h = (int)hidden.Shape[2];
            int w = (int)hidden.Shape[3];

            TensorShape upShape = new TensorShape(batch, ch, h * 2, w * 2);
            Tensor upsampled = new Tensor(upShape, hidden.DType);
            backend.UpsampleNearest2D(upsampled, hidden, 2, 2);
            hidden.Dispose();

            Tensor convUp = new Tensor(upShape, upsampled.DType);
            backend.Conv2D(convUp, upsampled, _upsampleWeight!, _upsampleBias, 1, 1, 1, 1);
            upsampled.Dispose();

            hidden = convUp;
        }

        return hidden;
    }

    /// <summary>Concatenates two tensors along the channel dimension (dim=1).</summary>
    private static Tensor ConcatChannels(IBackend backend, Tensor a, Tensor b)
    {
        int batch = (int)a.Shape[0];
        int chA = (int)a.Shape[1];
        int chB = (int)b.Shape[1];
        int height = (int)a.Shape[2];
        int width = (int)a.Shape[3];

        TensorShape outShape = new TensorShape(batch, chA + chB, height, width);
        Tensor output = new Tensor(outShape, a.DType);
        backend.Concat(output, [a, b], 1);

        return output;
    }
}
