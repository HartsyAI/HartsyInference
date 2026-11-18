using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Layers;

/// <summary>A scalar per row (a diffusion timestep, an interval, a guidance weight) embedded as <c>[cos, sin]</c> of
/// <c>value * exp(-ln(10000) k / (F/2))</c>, then <c>Linear(F, H)</c>, SiLU, <c>Linear(H, H)</c>, both without bias.</summary>
internal sealed unsafe class ScalarEmbed(int hiddenSize, int frequencyDim = ScalarEmbed.DefaultFrequencyDim)
{
    /// <summary>Width of the sinusoidal features in every checkpoint that uses this embedder.</summary>
    internal const int DefaultFrequencyDim = 256;

    private Tensor? _fc1;
    private Tensor? _fc2;

    /// <summary>Takes the two weights as stored (<c>[H, F]</c> and <c>[H, H]</c>); the tensors stay owned by the caller.</summary>
    internal void LoadWeights(Tensor fc1, Tensor fc2)
    {
        ArgumentNullException.ThrowIfNull(fc1);
        ArgumentNullException.ThrowIfNull(fc2);
        if (fc1.Shape.Rank != 2 || fc1.Shape[0] != hiddenSize || fc1.Shape[1] != frequencyDim)
        {
            throw new ArgumentException($"fc1 must be [{hiddenSize}, {frequencyDim}], got {fc1.Shape}.", nameof(fc1));
        }
        if (fc2.Shape.Rank != 2 || fc2.Shape[0] != hiddenSize || fc2.Shape[1] != hiddenSize)
        {
            throw new ArgumentException($"fc2 must be [{hiddenSize}, {hiddenSize}], got {fc2.Shape}.", nameof(fc2));
        }
        _fc1 = fc1;
        _fc2 = fc2;
    }

    /// <summary>One embedding row per value, as a fresh <c>[1, N, H]</c> tensor the caller owns.</summary>
    internal Tensor Forward(IBackend backend, ReadOnlySpan<float> values)
    {
        if (_fc1 is null || _fc2 is null)
        {
            throw new InvalidOperationException("ScalarEmbed weights are not loaded.");
        }
        int n = values.Length;
        Tensor features = new(new TensorShape(1, n, frequencyDim), DType.F32);
        FillFeatures((float*)features.DataPointer, values, frequencyDim);
        Tensor hidden = new(new TensorShape(1, n, hiddenSize), DType.F32);
        backend.Linear(hidden, features, _fc1, null);
        features.Dispose();
        Tensor active = new(hidden.Shape, DType.F32);
        backend.Silu(active, hidden);
        hidden.Dispose();
        Tensor result = new(active.Shape, DType.F32);
        backend.Linear(result, active, _fc2, null);
        active.Dispose();
        return result;
    }

    internal IEnumerable<Tensor> EnumerateWeights()
    {
        if (_fc1 is not null)
        {
            yield return _fc1;
        }
        if (_fc2 is not null)
        {
            yield return _fc2;
        }
    }

    /// <summary>Row-major <c>[N, F]</c>: the cosines of the F/2 frequencies, then their sines.</summary>
    internal static void FillFeatures(float* destination, ReadOnlySpan<float> values, int frequencyDim)
    {
        int half = frequencyDim / 2;
        const float maxPeriod = 10_000f;
        for (int i = 0; i < values.Length; i++)
        {
            int row = i * frequencyDim;
            for (int k = 0; k < half; k++)
            {
                float arg = values[i] * MathF.Exp(-MathF.Log(maxPeriod) * k / half);
                destination[row + k] = MathF.Cos(arg);
                destination[row + half + k] = MathF.Sin(arg);
            }
        }
    }
}
