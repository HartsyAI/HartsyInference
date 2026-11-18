using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>Maps one audio stream's normalized latents into the language model's embedding space:
/// <c>fc2(RMSNorm(fc1(x)))</c>, <c>fc1</c> and <c>fc2</c> with biases and the norm with eps 1e-6.</summary>
internal sealed class HaruAudioProjector : IDisposable
{
    private const float NormEps = 1e-6f;

    private readonly int _latentDim;
    private readonly int _hidden;
    private readonly Tensor _fc1;
    private readonly Tensor _fc1Bias;
    private readonly Tensor _norm;
    private readonly Tensor _fc2;
    private readonly Tensor _fc2Bias;
    private readonly List<Tensor> _owned = [];

    /// <summary>Reads <c>{prefix}.fc1.weight</c>, <c>.fc1.bias</c>, <c>.norm.weight</c>, <c>.fc2.weight</c> and <c>.fc2.bias</c>; the
    /// linears stay as stored, the biases and the norm weight are widened to F32 here. The checkpoint's tensors must outlive it.</summary>
    internal HaruAudioProjector(IReadOnlyDictionary<string, Tensor> weights, string prefix, int latentDim, int hidden)
    {
        _latentDim = latentDim;
        _hidden = hidden;
        _fc1 = Matrix(weights, $"{prefix}.fc1.weight", hidden, latentDim);
        _fc1Bias = Vector(weights, $"{prefix}.fc1.bias", hidden);
        _norm = Vector(weights, $"{prefix}.norm.weight", hidden);
        _fc2 = Matrix(weights, $"{prefix}.fc2.weight", hidden, hidden);
        _fc2Bias = Vector(weights, $"{prefix}.fc2.bias", hidden);
    }

    /// <summary>Projects <paramref name="latents"/> (<c>[1, T, latentDim]</c>) to a fresh <c>[1, T, hidden]</c> tensor the caller owns.</summary>
    internal Tensor Forward(IBackend backend, Tensor latents)
    {
        if (latents.Shape.Rank != 3 || latents.Shape[2] != _latentDim)
        {
            throw new ArgumentException($"latents must be [1, T, {_latentDim}], got {latents.Shape}.", nameof(latents));
        }
        long rows = latents.Shape[1];
        Tensor first = new(new TensorShape(1, rows, _hidden), DType.F32);
        backend.Linear(first, latents, _fc1, _fc1Bias);
        Tensor normed = new(first.Shape, DType.F32);
        backend.RmsNorm(normed, first, _norm, NormEps);
        first.Dispose();
        Tensor result = new(normed.Shape, DType.F32);
        backend.Linear(result, normed, _fc2, _fc2Bias);
        normed.Dispose();
        return result;
    }

    internal IEnumerable<Tensor> EnumerateWeights() => [_fc1, _fc1Bias, _norm, _fc2, _fc2Bias];

    public void Dispose()
    {
        foreach (Tensor t in _owned)
        {
            t.Dispose();
        }
        _owned.Clear();
    }

    private static Tensor Matrix(IReadOnlyDictionary<string, Tensor> weights, string key, int rows, int cols)
    {
        if (!weights.TryGetValue(key, out Tensor? t))
        {
            throw new KeyNotFoundException($"The Haru checkpoint has no tensor '{key}'.");
        }
        if (t.Shape.Rank != 2 || t.Shape[0] != rows || t.Shape[1] != cols)
        {
            throw new InvalidDataException($"'{key}' must be [{rows}, {cols}], got {t.Shape}.");
        }
        return t;
    }

    private Tensor Vector(IReadOnlyDictionary<string, Tensor> weights, string key, int size)
    {
        if (!weights.TryGetValue(key, out Tensor? t))
        {
            throw new KeyNotFoundException($"The Haru checkpoint has no tensor '{key}'.");
        }
        if (t.ElementCount != size)
        {
            throw new InvalidDataException($"'{key}' must hold {size} values, it has {t.ElementCount}.");
        }
        if (t.DType == DType.F32)
        {
            return t;
        }
        Tensor widened = t.CastTo(DType.F32);
        _owned.Add(widened);
        return widened;
    }
}
