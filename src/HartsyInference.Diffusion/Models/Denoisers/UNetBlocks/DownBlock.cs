using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;

/// <summary>UNet down block: sequence of (ResNet + optional CrossAttention) layers followed by an optional downsample Conv2d(stride=2).</summary>
public sealed class DownBlock
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _numLayers;
    private readonly bool _hasAttention;
    private readonly bool _hasDownsample;

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

    // Downsample: Conv2d(outCh, outCh, 3, stride=2, padding=1)
    private Tensor? _downsampleWeight;
    private Tensor? _downsampleBias;

    /// <summary>Creates a UNet down block.</summary>
    public DownBlock(int inChannels, int outChannels, int timeDim, int numLayers, bool hasAttention, bool hasDownsample, int numHeads = 8, int crossAttentionDim = 768, int numTransformerBlocks = 1)
    {
        _inChannels = inChannels;
        _outChannels = outChannels;
        _numLayers = numLayers;
        _hasAttention = hasAttention;
        _hasDownsample = hasDownsample;

        _resnets = new UNetResNetBlock[numLayers];
        _attentions = new CrossAttentionBlock?[numLayers];

        for (int i = 0; i < numLayers; i++)
        {
            int resInCh = i == 0 ? inChannels : outChannels;
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

        if (_hasDownsample)
        {
            _downsampleWeight = weights[$"{prefix}.downsamplers.0.conv.weight"];
            _downsampleBias = weights[$"{prefix}.downsamplers.0.conv.bias"];
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
        if (_downsampleWeight is not null) yield return _downsampleWeight;
        if (_downsampleBias is not null) yield return _downsampleBias;
    }

    /// <summary>Forward pass without IPA injection. Returns (output, skipConnections).</summary>
    public (Tensor output, List<Tensor> skips) Forward(IBackend backend, Tensor input, Tensor temb, Tensor context)
    {
        return Forward(backend, input, temb, context, ipaImageTokens: null, ipaToKIpAll: null, ipaToVIpAll: null, ipaStartIndex: 0, ipaScalePerLayer: null);
    }

    /// <summary>Forward pass with optional IP-Adapter image-attention injection. The block consumes a contiguous range of the flat per-cross-attn-layer K/V lists starting at <paramref name="ipaStartIndex"/> and length <see cref="CrossAttentionLayerCount"/>; each <see cref="CrossAttentionBlock"/> internally consumes its own <see cref="CrossAttentionBlock.NumTransformerBlocks"/> entries. Per-layer scale comes from <paramref name="ipaScalePerLayer"/> at the same flat indices.</summary>
    public (Tensor output, List<Tensor> skips) Forward(IBackend backend, Tensor input, Tensor temb, Tensor context,
        Tensor? ipaImageTokens, IReadOnlyList<Tensor>? ipaToKIpAll, IReadOnlyList<Tensor>? ipaToVIpAll, int ipaStartIndex, IReadOnlyList<float>? ipaScalePerLayer)
    {
        List<Tensor> skips = new List<Tensor>();
        Tensor hidden = input;
        int ipaOffset = ipaStartIndex;

        for (int i = 0; i < _numLayers; i++)
        {
            Tensor resOut = _resnets[i].Forward(backend, hidden, temb);
            if (hidden != input) hidden.Dispose();
            hidden = resOut;

            if (_hasAttention && _attentions[i] is not null)
            {
                Tensor attnOut = _attentions[i]!.Forward(backend, hidden, context,
                    ipaImageTokens, ipaToKIpAll, ipaToVIpAll, ipaOffset, ipaScalePerLayer);
                ipaOffset += _attentions[i]!.NumTransformerBlocks;
                hidden.Dispose();
                hidden = attnOut;
            }

            // Save skip connection (clone so we own it)
            skips.Add(hidden.To(hidden.Device));
        }

        // Downsample
        if (_hasDownsample)
        {
            int batch = (int)hidden.Shape[0];
            int ch = (int)hidden.Shape[1];
            int h = (int)hidden.Shape[2];
            int w = (int)hidden.Shape[3];

            TensorShape downShape = new TensorShape(batch, ch, h / 2, w / 2);
            Tensor downsampled = new Tensor(downShape, hidden.DType);
            backend.Conv2D(downsampled, hidden, _downsampleWeight!, _downsampleBias, 2, 2, 1, 1);
            hidden.Dispose();
            hidden = downsampled;

            // Save downsample output as skip (diffusers does this)
            skips.Add(hidden.To(hidden.Device));
        }

        return (hidden, skips);
    }
}
