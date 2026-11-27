using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Fusion encoder stack in which image tokens attend to the text prompt; the prompt is not updated.</summary>
public sealed class Sam3FusionEncoder
{
    private readonly Sam3FusionEncoderLayer[] _layers;

    /// <summary>Creates the encoder stack from the configuration.</summary>
    public Sam3FusionEncoder(Sam3Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.EncoderLayers < 1)
            throw new ArgumentException("The fusion encoder needs at least one layer.", nameof(config));
        _layers = new Sam3FusionEncoderLayer[config.EncoderLayers];
        for (int i = 0; i < _layers.Length; i++)
            _layers[i] = new Sam3FusionEncoderLayer(config.DModel, config.EncoderHeads, config.EncoderFfn, config.LayerNormEps);
    }

    /// <summary>Loads <c>layers.{i}.*</c> for every layer; fails naming the first missing key.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(weights, $"layers.{i}.");
    }

    /// <summary>Fuses <paramref name="imageTokens"/> <c>[1, HW, D]</c> with <paramref name="prompt"/> <c>[1, T, D]</c>.
    /// <paramref name="imagePos"/> is the image position encoding of the same shape. Returns a new <c>[1, HW, D]</c>
    /// tensor the caller owns; the inputs are not disposed.</summary>
    public Tensor Forward(IBackend backend, Tensor imageTokens, Tensor imagePos, Tensor prompt, bool[] promptPadding)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(imageTokens);
        ArgumentNullException.ThrowIfNull(imagePos);
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(promptPadding);

        Tensor current = _layers[0].Forward(backend, imageTokens, imagePos, prompt, null, promptPadding);
        for (int i = 1; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, current, imagePos, prompt, null, promptPadding);
            current.Dispose();
            current = next;
        }
        return current;
    }
}
