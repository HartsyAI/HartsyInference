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

        // Finite values only: MathF.Max propagates NaN, so one legitimately-NaN element would make peak NaN, then
        // limit NaN, and then every `delta <= limit` false — the comparison would fail on tensors that agree.
        float peak = 0f;
        foreach (float value in e)
        {
            if (float.IsFinite(value))
            {
                peak = MathF.Max(peak, MathF.Abs(value));
            }
        }
        float limit = atol + rtol * peak;

        // A non-finite value has to be caught by identity, never by magnitude. NaN compares false to everything, so
        // `delta > limit` is false when a kernel produces NaN against a finite reference — the comparison reports
        // success for precisely the failure it exists to catch. Infinity is the same story one step later: it is a
        // real divergence that a relative tolerance scaled by a finite peak would have to call enormous, and the
        // arithmetic below is not the place to decide that.
        for (long i = 0; i < a.LongLength; i++)
        {
            if (float.IsFinite(a[i]) == float.IsFinite(e[i]) && (float.IsFinite(a[i]) || a[i].Equals(e[i])))
            {
                continue;
            }
            throw new InvalidOperationException(
                $"Tensors disagree on a non-finite value at [{i}]: actual {a[i]:G9}, expected {e[i]:G9}. "
                + $"No tolerance applies — a NaN compares false to every bound, so this would otherwise "
                + $"pass.{Suffix(because)}");
        }

        long worstIndex = -1;
        float worstDelta = 0f;
        long differing = 0;
        for (long i = 0; i < a.LongLength; i++)
        {
            // Proven equal by the pass above, and subtracting them would only produce a NaN delta.
            if (!float.IsFinite(a[i]))
            {
                continue;
            }
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

        if (worstIndex < 0 || worstDelta <= limit)
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
    /// <remarks>Compares dtype and raw storage, not values. Going through <see cref="Close"/> with a zero tolerance
    /// looks equivalent and is not: it widens both sides to F32 first, so an F16 result matches an F32 one, and it
    /// subtracts, so negative zero matches positive zero and two different NaN payloads match each other. Those are
    /// representation changes, and representation is the thing this overload is asked about.</remarks>
    public static void Identical(Tensor actual, Tensor expected, string? because = null)
    {
        ArgumentNullException.ThrowIfNull(actual);
        ArgumentNullException.ThrowIfNull(expected);
        if (!actual.Shape.Equals(expected.Shape))
        {
            throw new InvalidOperationException(
                $"Shape mismatch: actual {actual.Shape}, expected {expected.Shape}.{Suffix(because)}");
        }
        if (actual.DType != expected.DType)
        {
            throw new InvalidOperationException(
                $"DType mismatch: actual {actual.DType.Name}, expected {expected.DType.Name}.{Suffix(because)}");
        }

        ReadOnlySpan<byte> a = actual.AsReadOnlySpan<byte>();
        ReadOnlySpan<byte> e = expected.AsReadOnlySpan<byte>();
        if (a.Length != e.Length)
        {
            throw new InvalidOperationException(
                $"Storage size mismatch: actual {a.Length} bytes, expected {e.Length} bytes.{Suffix(because)}");
        }
        if (a.SequenceEqual(e))
        {
            return;
        }

        long stride = Math.Max(actual.DType.ComputeByteCount(1), 1);
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != e[i])
            {
                throw new InvalidOperationException(
                    $"Tensors differ at byte {i} (element {i / stride}): actual 0x{a[i]:X2}, "
                    + $"expected 0x{e[i]:X2}.{Suffix(because)}");
            }
        }
    }

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
