using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Position-wise feed-forward block (Linear → ReLU → Linear) used by both the feature enhancer and the
/// decoder. ReLU (Grounding DINO's activation) is expressed as <c>LeakyRelu(slope=0)</c> since the backend has no
/// dedicated ReLU op. Returns the block output; the caller applies the residual + norm.</summary>
public sealed class GroundingDinoFeedForward
{
    private readonly int _dim;
    private readonly int _inner;

    private Tensor? _fc1W, _fc1B, _fc2W, _fc2B;

    public GroundingDinoFeedForward(int dim, int inner)
    {
        _dim = dim;
        _inner = inner;
    }

    public void Build(IGroundingWeightBag bag, string prefix)
    {
        _fc1W = bag.Get($"{prefix}.fc1.weight", _inner, _dim);
        _fc1B = bag.Get($"{prefix}.fc1.bias", _inner);
        _fc2W = bag.Get($"{prefix}.fc2.weight", _dim, _inner);
        _fc2B = bag.Get($"{prefix}.fc2.bias", _dim);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_fc1W, _fc1B, _fc2W, _fc2B];
        foreach (Tensor? t in all)
            if (t is not null) yield return t;
    }

    /// <summary>Runs the block on <paramref name="x"/> <c>[1, S, dim]</c>. Caller owns the result.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        if (_fc1W is null)
            throw new InvalidOperationException("GroundingDinoFeedForward weights not loaded.");
        int s = (int)x.Shape[1];
        Tensor hidden = new Tensor(new TensorShape(1, s, _inner), DType.F32);
        backend.Linear(hidden, x, _fc1W!, _fc1B);
        backend.LeakyRelu(hidden, hidden, 0f);
        Tensor output = new Tensor(new TensorShape(1, s, _dim), DType.F32);
        backend.Linear(output, hidden, _fc2W!, _fc2B);
        hidden.Dispose();
        return output;
    }
}
