using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.UNetBlocks;

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

    /// <summary>Forward pass. Returns (output, skipConnections). Each ResNet/Attention output is saved as a skip connection for the corresponding up block.</summary>
    public (Tensor output, List<Tensor> skips) Forward(IBackend backend, Tensor input, Tensor temb, Tensor context)
    {
        List<Tensor> skips = new List<Tensor>();
        Tensor hidden = input;

        for (int i = 0; i < _numLayers; i++)
        {
            Tensor resOut = _resnets[i].Forward(backend, hidden, temb);
            if (hidden != input) hidden.Dispose();
            hidden = resOut;

            if (_hasAttention && _attentions[i] is not null)
            {
                Tensor attnOut = _attentions[i]!.Forward(backend, hidden, context);
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
            Tensor downsampled = new Tensor(downShape, DType.F32);
            backend.Conv2D(downsampled, hidden, _downsampleWeight!, _downsampleBias, 2, 2, 1, 1);
            hidden.Dispose();
            hidden = downsampled;

            // Save downsample output as skip (diffusers does this)
            skips.Add(hidden.To(hidden.Device));
        }

        return (hidden, skips);
    }
}
