using System.Linq;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cpu.Moe;

/// <summary>
/// The CPU share of a planned layer on packed weights: each expert's Q8_0 or Q4_K projections are read in place by
/// <see cref="CpuExpertKernels"/>, with no F32 copy. <paramref name="resolve"/> supplies an expert's packed tensors; they must
/// stay alive while the expert runs, as a pack reader's views do while the reader is open.
/// </summary>
public sealed class PackedExpertHostRunner : IExpertHostRunner
{
    private readonly DType _dtype;
    private readonly int _hidden;
    private readonly int _intermediate;
    private readonly Func<ExpertKey, ExpertWeights> _resolve;

    /// <summary>Creates a runner for experts of one packed dtype and shape.</summary>
    /// <param name="dtype">The packed dtype of every projection: <see cref="DType.Q8_0"/> or <see cref="DType.Q4_K"/>.</param>
    /// <param name="hidden">Model width H.</param>
    /// <param name="intermediate">Expert inner width I.</param>
    /// <param name="resolve">Supplies an expert's packed projections; called once per run.</param>
    /// <exception cref="NotSupportedException">The dtype has no packed CPU kernel.</exception>
    public PackedExpertHostRunner(DType dtype, int hidden, int intermediate, Func<ExpertKey, ExpertWeights> resolve)
    {
        ArgumentNullException.ThrowIfNull(resolve);
        if (dtype != DType.Q8_0 && dtype != DType.Q4_K)
            throw new NotSupportedException($"{dtype.Name} has no packed CPU kernel; use Q8_0 or Q4_K.");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hidden);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(intermediate);
        _dtype = dtype;
        _hidden = hidden;
        _intermediate = intermediate;
        _resolve = resolve;
    }

    /// <inheritdoc/>
    /// <remarks>Runs in chunks of at most <see cref="CpuExpertKernels.MaxRows"/> rows, the kernel's limit.</remarks>
    /// <exception cref="InvalidOperationException">The resolved expert's projections are not this runner's dtype, or the
    /// expert carries companion tensors, which this runner does not read.</exception>
    public void Run(ExpertProgram program, ExpertKey key, ReadOnlySpan<float> x, int rows, Span<float> y)
    {
        ExpertWeights weights = _resolve(key);
        if (weights.W1.Companions.Any() || weights.W2.Companions.Any() || weights.W3.Companions.Any())
            throw new InvalidOperationException($"{key} carries scale or bias tensors this runner does not read; use the F32 reference.");
        ReadOnlySpan<byte> gate = PackedBytes(weights.W1, key);
        ReadOnlySpan<byte> up = PackedBytes(weights.W3, key);
        ReadOnlySpan<byte> down = PackedBytes(weights.W2, key);
        int hidden = _hidden;
        for (int first = 0; first < rows; first += CpuExpertKernels.MaxRows)
        {
            int count = Math.Min(CpuExpertKernels.MaxRows, rows - first);
            CpuExpertKernels.Apply(program, _dtype, _hidden, _intermediate, gate, up, down,
                x.Slice(first * hidden, count * hidden), count, y.Slice(first * hidden, count * hidden));
        }
        // The spans point into native memory the tensors own; the weights must not be collected before the kernel returns.
        GC.KeepAlive(weights);
    }

    private unsafe ReadOnlySpan<byte> PackedBytes(ExpertMatrix matrix, ExpertKey key)
    {
        Tensor weight = matrix.Weight;
        if (weight.DType != _dtype)
            throw new InvalidOperationException($"{key} holds {weight.DType.Name} weights; this runner serves {_dtype.Name}.");
        return new ReadOnlySpan<byte>(weight.DataPointer, checked((int)_dtype.ComputeByteCount(weight.ElementCount)));
    }
}
