using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// The upstream <c>TimestepEmbedder</c> (<c>timestep_embedder.py</c>): a sinusoid of the scalar (cosines first, then sines, <c>max_period</c> 10000)
/// through <c>Linear(frequency, hidden)</c>, SiLU and <c>Linear(hidden, hidden)</c>. The structure flow uses one for the timestep and one for the
/// shortcut step size <c>d</c>. Keys: <c>mlp.0.{weight,bias}</c> and <c>mlp.2.{weight,bias}</c> under the prefix; the weights stay owned by the caller.
/// </summary>
public sealed class SamTimestepEmbedder
{
    private const float MaxPeriod = 10000f;

    private readonly Tensor _w1;
    private readonly Tensor _b1;
    private readonly Tensor _w2;
    private readonly Tensor _b2;

    public SamTimestepEmbedder(IReadOnlyDictionary<string, Tensor> weights, string prefix, int hidden, int frequencySize = 256)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(prefix);
        if (hidden <= 0 || frequencySize <= 0)
            throw new ArgumentException("The hidden and frequency widths must be positive.");
        Hidden = hidden;
        FrequencySize = frequencySize;
        _w1 = SamWeights.Require(weights, prefix + "mlp.0.weight", hidden, frequencySize);
        _b1 = SamWeights.Require(weights, prefix + "mlp.0.bias", hidden);
        _w2 = SamWeights.Require(weights, prefix + "mlp.2.weight", hidden, hidden);
        _b2 = SamWeights.Require(weights, prefix + "mlp.2.bias", hidden);
    }

    public int Hidden { get; }

    public int FrequencySize { get; }

    /// <summary>The sinusoid of <paramref name="t"/>: <c>cos(t · f_i)</c> then <c>sin(t · f_i)</c> with <c>f_i = exp(-ln(10000) · i / half)</c>, and a
    /// trailing zero when the size is odd.</summary>
    public static void Sinusoid(float t, Span<float> destination)
    {
        int half = destination.Length / 2;
        float logMax = MathF.Log(MaxPeriod);
        for (int i = 0; i < half; i++)
        {
            float angle = t * MathF.Exp(-logMax * i / half);
            destination[i] = MathF.Cos(angle);
            destination[half + i] = MathF.Sin(angle);
        }

        if (destination.Length % 2 == 1)
            destination[^1] = 0f;
    }

    /// <summary>The embedding of one scalar: F32 <c>[1, hidden]</c>, owned by the caller.</summary>
    public Tensor Embed(IBackend backend, float t)
    {
        ArgumentNullException.ThrowIfNull(backend);
        using Tensor sinusoid = new(new TensorShape(1, FrequencySize), DType.F32);
        Sinusoid(t, sinusoid.AsSpan<float>());
        using Tensor first = new(new TensorShape(1, Hidden), DType.F32);
        backend.Linear(first, sinusoid, _w1, _b1);
        using Tensor activated = new(first.Shape, DType.F32);
        backend.Silu(activated, first);
        Tensor result = new(new TensorShape(1, Hidden), DType.F32);
        backend.Linear(result, activated, _w2, _b2);
        return result;
    }
}
