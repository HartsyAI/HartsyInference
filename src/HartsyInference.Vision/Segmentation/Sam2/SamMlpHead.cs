using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam2;

/// <summary>SAM's small <c>MLP</c> head (used for the mask hypernetworks and the IoU-prediction head): a stack
/// of <c>num_layers</c> linear layers with ReLU between them (none after the last), keyed <c>layers.{i}.*</c>.
/// The layer count is discovered from the weights so both 3-layer heads load without configuration.</summary>
public sealed class SamMlpHead
{
    private readonly List<Tensor> _weights = new();
    private readonly List<Tensor?> _biases = new();

    /// <summary>Loads consecutive <c>{prefix}layers.{i}.weight/bias</c> until a gap. Requires at least one layer.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        int i = 0;
        while (weights.TryGetValue($"{prefix}layers.{i}.weight", out Tensor? w))
        {
            _weights.Add(w.DType != DType.F32 ? w.CastTo(DType.F32) : w);
            _biases.Add(weights.TryGetValue($"{prefix}layers.{i}.bias", out Tensor? b)
                ? (b.DType != DType.F32 ? b.CastTo(DType.F32) : b) : null);
            i++;
        }
        if (_weights.Count == 0)
            throw new KeyNotFoundException($"SamMlpHead: no layers found under '{prefix}layers.*'.");
    }

    /// <summary>Yields every loaded weight/bias tensor.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _weights) yield return w;
        foreach (Tensor? b in _biases)
            if (b is not null) yield return b;
    }

    /// <summary>Runs the head on <paramref name="x"/> <c>[M, inDim]</c>; returns <c>[M, outDim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        if (_weights.Count == 0)
            throw new InvalidOperationException("SamMlpHead weights not loaded.");

        int m = (int)x.Shape[0];
        Tensor cur = x;
        for (int i = 0; i < _weights.Count; i++)
        {
            int outDim = (int)_weights[i].Shape[0];
            Tensor next = new Tensor(new TensorShape(m, outDim), DType.F32);
            backend.Linear(next, cur, _weights[i], _biases[i]);
            if (i < _weights.Count - 1)
                backend.Clamp(next, next, 0f, float.MaxValue); // ReLU
            cur = next;
        }
        return cur;
    }
}
