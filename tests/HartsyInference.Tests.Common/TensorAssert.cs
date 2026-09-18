using HartsyInference.Core.Tensors;

namespace HartsyInference.Tests.Common;

/// <summary>Compares two tensors and, when they differ, says where and by how much.
///
/// <para>Written because the per-op comparison in these suites is hand-rolled each time — a loop, a literal
/// tolerance, and an assert that reports "expected true, got false". That tells you a kernel is wrong and nothing
/// else. A backend comparison needs the worst index and both values at it, because the shape of the error is the
/// diagnosis: one bad element is an indexing bug, a whole row is a stride bug, and a small relative error
/// everywhere is just a different accumulation order.</para></summary>
public static class TensorAssert
{
    /// <summary>Asserts every element matches within <paramref name="rtol"/> relative to the expected tensor's own
    /// scale, plus <paramref name="atol"/> absolute.</summary>
    /// <remarks>Relative to the tensor's peak magnitude rather than per-element: a per-element relative tolerance is
    /// meaningless where the expected value is near zero, which is most of a normalized activation.</remarks>
    public static void Close(Tensor actual, Tensor expected, float rtol = 1e-4f, float atol = 1e-5f, string? because = null)
    {
        ArgumentNullException.ThrowIfNull(actual);
        ArgumentNullException.ThrowIfNull(expected);
        if (!actual.Shape.Equals(expected.Shape))
        {
            throw new InvalidOperationException(
                $"Shape mismatch: actual {actual.Shape}, expected {expected.Shape}.{Suffix(because)}");
        }

        float[] a = ToF32(actual);
        float[] e = ToF32(expected);

        float peak = 0f;
        foreach (float value in e)
        {
            peak = MathF.Max(peak, MathF.Abs(value));
        }
        float limit = atol + rtol * peak;

        long worstIndex = -1;
        float worstDelta = 0f;
        long differing = 0;
        for (long i = 0; i < a.LongLength; i++)
        {
            float delta = MathF.Abs(a[i] - e[i]);
            if (delta > limit)
            {
                differing++;
            }
            if (delta > worstDelta)
            {
                worstDelta = delta;
                worstIndex = i;
            }
        }

        if (worstDelta <= limit)
        {
            return;
        }

        throw new InvalidOperationException(
            $"Tensors differ at {differing} of {a.LongLength} elements. Worst at [{worstIndex}]: "
            + $"actual {a[worstIndex]:G9}, expected {e[worstIndex]:G9}, delta {worstDelta:E3} "
            + $"(limit {limit:E3} = atol {atol:E3} + rtol {rtol:E3} x peak {peak:E3}).{Suffix(because)}");
    }

    /// <summary>Asserts the two tensors are bit-identical. For a refactor that must not change a result at all:
    /// a tolerance would hide exactly the drift such a change is being checked for.</summary>
    public static void Identical(Tensor actual, Tensor expected, string? because = null)
        => Close(actual, expected, rtol: 0f, atol: 0f, because);

    private static float[] ToF32(Tensor tensor)
    {
        // Comparison is in F32 whatever the storage is: the question is whether the VALUES agree, and two backends
        // may legitimately hold the same value in different widths.
        if (tensor.DType == DType.F32)
        {
            return tensor.AsReadOnlySpan<float>().ToArray();
        }
        // EnsureF32 returns the tensor itself when it is already F32, so only the converted copy is ours to dispose.
        Tensor widened = TensorCasts.EnsureF32(tensor);
        try
        {
            return widened.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            if (!ReferenceEquals(widened, tensor))
            {
                widened.Dispose();
            }
        }
    }

    private static string Suffix(string? because) => string.IsNullOrEmpty(because) ? "" : $" {because}";
}
