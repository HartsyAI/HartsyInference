using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>A small multi-layer perceptron (the RT-DETR <c>RTDetrMLPPredictionHead</c>): a stack of
/// linear layers with ReLU between them and no activation on the last. Used for the query position
/// head (4→2d→d), the encoder box head, and the per-layer box refinement heads (d→d→d→4).</summary>
public sealed class RtDetrMlp
{
    private readonly RtDetrLinear[] _layers;

    /// <summary>Creates an MLP with the given per-layer output feature counts.</summary>
    public RtDetrMlp(IReadOnlyList<int> outFeatures)
    {
        ArgumentNullException.ThrowIfNull(outFeatures);
        _layers = new RtDetrLinear[outFeatures.Count];
        for (int i = 0; i < outFeatures.Count; i++)
            _layers[i] = new RtDetrLinear(outFeatures[i]);
    }

    /// <summary>Loads <c>{prefix}.layers.{i}</c> for each layer.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(weights, $"{prefix}.layers.{i}");
    }

    /// <summary>Yields every layer's weight + bias for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (RtDetrLinear l in _layers)
            foreach (Tensor t in l.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs the MLP with ReLU between layers (none after the last). Owns the returned tensor;
    /// does not consume <paramref name="input"/>.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        Tensor x = _layers[0].Forward(backend, input);
        for (int i = 1; i < _layers.Length; i++)
        {
            Relu(x);
            Tensor next = _layers[i].Forward(backend, x);
            x.Dispose();
            x = next;
        }
        return x;
    }

    private static void Relu(Tensor t)
    {
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++)
            if (s[i] < 0f) s[i] = 0f;
    }
}
