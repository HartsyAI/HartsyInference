using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity;

/// <summary>Outcome of comparing one output tensor against its expectation.</summary>
public sealed record CompareResult(string Name, bool Pass, string Detail, double MaxAbs, double MaxRel);

/// <summary>Tensor comparison under a case's acceptance rule.</summary>
public static class TensorCompare
{
    /// <summary>Compares <paramref name="actual"/> to <paramref name="expected"/>: shape and dtype must match; then bytes, or tolerances.</summary>
    public static CompareResult Compare(string name, Tensor expected, Tensor actual, Acceptance acceptance, bool mayBeNonFinite)
    {
        if (expected.Shape.ToString() != actual.Shape.ToString() || expected.DType != actual.DType)
            return new(name, false, $"shape/dtype {actual.Shape} {actual.DType} vs expected {expected.Shape} {expected.DType}", double.NaN, double.NaN);
        if (acceptance.Kind == AcceptanceKind.BitIdentical)
        {
            bool same = expected.AsReadOnlySpan<byte>().SequenceEqual(actual.AsReadOnlySpan<byte>());
            if (!mayBeNonFinite && !AllFinite(actual, out string where))
                return new(name, false, "non-finite value " + where, double.NaN, double.NaN);
            if (same) return new(name, true, "bit-identical", 0, 0);
            (double byteAbs, double byteRel) = Deviation(expected, actual);
            return new(name, false, double.IsNaN(byteAbs) ? "bytes differ" : $"bytes differ: maxAbs {byteAbs:E2} maxRel {byteRel:E2}", byteAbs, byteRel);
        }
        using Tensor e32 = expected.DType == DType.F32 ? expected : expected.CastTo(DType.F32);
        using Tensor a32 = actual.DType == DType.F32 ? actual : actual.CastTo(DType.F32);
        ReadOnlySpan<float> e = e32.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = a32.AsReadOnlySpan<float>();
        double maxAbs = 0, maxRel = 0;
        for (int i = 0; i < e.Length; i++)
        {
            float ev = e[i], av = a[i];
            if (!float.IsFinite(av))
            {
                if (!mayBeNonFinite) return new(name, false, $"non-finite value at [{i}]", double.NaN, double.NaN);
                if (float.IsNaN(ev) && float.IsNaN(av)) continue;
                if (ev == av) continue;
                return new(name, false, $"non-finite mismatch at [{i}]: {av} vs {ev}", double.NaN, double.NaN);
            }
            double diff = Math.Abs((double)av - ev);
            maxAbs = Math.Max(maxAbs, diff);
            maxRel = Math.Max(maxRel, diff / Math.Max(Math.Abs((double)ev), 1e-6));
        }
        bool pass = maxAbs <= acceptance.MaxAbs || maxRel <= acceptance.MaxRel;
        return new(name, pass, $"maxAbs {maxAbs:E2} maxRel {maxRel:E2} (limits {acceptance.MaxAbs:E1} / {acceptance.MaxRel:E1})", maxAbs, maxRel);
    }

    /// <summary>The largest absolute and relative deviation between two float tensors (F32, F16 or BF16), or NaN for other dtypes
    /// and for a non-finite element: the number a bit mismatch is worth knowing.</summary>
    private static (double MaxAbs, double MaxRel) Deviation(Tensor expected, Tensor actual)
    {
        if (expected.DType != DType.F32 && expected.DType != DType.F16 && expected.DType != DType.BF16) return (double.NaN, double.NaN);
        using Tensor e32 = expected.DType == DType.F32 ? expected : expected.CastTo(DType.F32);
        using Tensor a32 = actual.DType == DType.F32 ? actual : actual.CastTo(DType.F32);
        ReadOnlySpan<float> e = e32.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = a32.AsReadOnlySpan<float>();
        double maxAbs = 0, maxRel = 0;
        for (int i = 0; i < e.Length; i++)
        {
            if (!float.IsFinite(e[i]) || !float.IsFinite(a[i])) return (double.NaN, double.NaN);
            double diff = Math.Abs((double)a[i] - e[i]);
            maxAbs = Math.Max(maxAbs, diff);
            maxRel = Math.Max(maxRel, diff / Math.Max(Math.Abs((double)e[i]), 1e-6));
        }
        return (maxAbs, maxRel);
    }

    private static bool AllFinite(Tensor t, out string where)
    {
        where = "";
        if (t.DType != DType.F32) return true;
        ReadOnlySpan<float> s = t.AsReadOnlySpan<float>();
        for (int i = 0; i < s.Length; i++)
        {
            if (!float.IsFinite(s[i])) { where = $"at [{i}]"; return false; }
        }
        return true;
    }
}
