using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Tests.Common.Attention;

/// <summary>Host values for one case, rounded to the case's dtype: the values the kernel actually sees. Layout follows the
/// case (head-major <c>[B,H,S,D]</c> or token-major <c>[S,H·D]</c>).</summary>
public sealed class AttnInputs
{
    /// <summary>Query values.</summary>
    public required float[] Q { get; init; }

    /// <summary>Key values (the whole cache for a flash case).</summary>
    public required float[] K { get; init; }

    /// <summary>Value values.</summary>
    public required float[] V { get; init; }

    /// <summary>F32 mask values in <see cref="AttnCase.MaskShape"/>, or null.</summary>
    public float[]? Mask { get; init; }

    /// <summary>Seeded inputs for <paramref name="c"/>; <paramref name="mutate"/> rewrites the generated arrays first.</summary>
    public static AttnInputs Create(AttnCase c, Action<float[], float[], float[], float[]?>? mutate = null)
    {
        Random rng = new(c.Seed);
        float[] q = Fill(rng, c.Batch * c.Heads * c.Sq * c.HeadDim);
        float[] k = Fill(rng, c.Batch * c.KvHeads * c.Skv * c.HeadDim);
        float[] v = Fill(rng, c.Batch * c.KvHeads * c.Skv * c.HeadDim);
        float[]? mask = null;
        if (c.MaskShape.Length > 0)
        {
            long[] s = c.MaskShape;
            long skv = s[^1], rows = s[^2], heads = s.Length >= 3 ? s[^3] : 1;
            mask = new float[s.Aggregate(1L, (a, x) => a * x)];
            for (long i = 0; i < mask.Length; i++)
            {
                long key = i % skv, row = i / skv % rows, head = i / skv / rows % heads, batch = i / skv / rows / heads;
                mask[i] = 0.7f * MathF.Sin(batch * 3f + head * 1.3f + row * 0.11f + key * 0.07f);
            }
            // A finite -1e30 row keeps its mass: the contract makes it uniform over every key, on every backend.
            if (c.Id.Contains("sentinel", StringComparison.Ordinal))
                for (long key = 0; key < skv; key++) mask[2 * skv + key] = -1e30f;
        }
        mutate?.Invoke(q, k, v, mask);
        Round(q, c.DType);
        Round(k, c.DType);
        Round(v, c.DType);
        return new AttnInputs { Q = q, K = k, V = v, Mask = mask };
    }

    private static float[] Fill(Random rng, int n)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static void Round(float[] a, AttnDType dtype)
    {
        if (dtype == AttnDType.F16)
            for (int i = 0; i < a.Length; i++) a[i] = (float)(Half)a[i];
        else if (dtype == AttnDType.BF16)
            for (int i = 0; i < a.Length; i++)
            {
                uint u = BitConverter.SingleToUInt32Bits(a[i]);
                a[i] = BitConverter.UInt32BitsToSingle(((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16) << 16);
            }
    }
}

/// <summary>Whether a backend ran a case through its legacy entry or refused it as that entry's contract says it must.</summary>
public enum AttnOutcome { Ran, Refused, Unsupported }

/// <summary>Runs a case through the legacy entry point it names and reads the output back as F32.</summary>
public static class AttnInvoker
{
    /// <summary>What <paramref name="kind"/>'s legacy entry is expected to do with <paramref name="c"/>.</summary>
    public static AttnOutcome Expected(string kind, AttnCase c) => kind switch
    {
        // Head-major SDPA takes K/V with Q's head count on CUDA and the CPU; grouped callers go token-major or flash.
        "cuda" when !c.Flash && c.Layout == AttnLayout.HeadMajor && c.KvHeads != c.Heads => AttnOutcome.Refused,
        "cpu" when !c.Flash && c.KvHeads != c.Heads => AttnOutcome.Refused,
        "cpu" when c.DType != AttnDType.F32 => AttnOutcome.Refused,
        _ => AttnOutcome.Ran,
    };

    /// <summary>Runs <paramref name="c"/> on <paramref name="backend"/>; <paramref name="output"/> is the F32 result when it ran.</summary>
    public static AttnOutcome Run(IBackend backend, AttnCase c, AttnInputs inputs, out float[]? output, out Exception? refusal)
    {
        output = null;
        refusal = null;
        if (c.Layout == AttnLayout.TokenMajor
            && !(c.KvHeads == c.Heads ? backend.SupportsTokenMajorAttention : backend.SupportsTokenMajorGqaAttention))
            return AttnOutcome.Unsupported;
        DType dt = c.DType switch { AttnDType.F16 => DType.F16, AttnDType.BF16 => DType.BF16, _ => DType.F32 };
        bool tm = c.Layout == AttnLayout.TokenMajor;
        TensorShape qs = tm ? new(c.Sq, c.Heads * c.HeadDim) : new(c.Batch, c.Heads, c.Sq, c.HeadDim);
        TensorShape ks = tm ? new(c.Skv, c.KvHeads * c.HeadDim) : new(c.Batch, c.KvHeads, c.Skv, c.HeadDim);
        using Tensor q = Load(qs, inputs.Q, dt), k = Load(ks, inputs.K, dt), v = Load(ks, inputs.V, dt);
        using Tensor? mask = inputs.Mask is null ? null : Load(new TensorShape(c.MaskShape), inputs.Mask, DType.F32);
        using Tensor o = new(qs, dt);
        try
        {
            if (c.Flash)
                backend.FlashAttention(o, q, k, v, c.KvLen, c.Group, c.Causal, c.QOffset, c.Scale, slidingWindow: c.Window);
            else if (tm && c.KvHeads == c.Heads)
                backend.ScaledDotProductAttentionTokenMajor(o, q, k, v, mask, c.Heads, c.HeadDim, c.Scale, c.AllowF16);
            else if (tm)
                backend.ScaledDotProductAttentionTokenMajor(o, q, k, v, mask, c.Heads, c.KvHeads, c.HeadDim, c.Scale, c.AllowF16);
            else
                backend.ScaledDotProductAttention(o, q, k, v, mask, c.Scale, c.AllowF16);
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException)
        {
            refusal = e;
            return AttnOutcome.Refused;
        }
        output = ReadF32(o);
        return AttnOutcome.Ran;
    }

    /// <summary>The tensor's values as F32, decoded from its storage bits.</summary>
    public static float[] ReadF32(Tensor t)
    {
        if (t.DType == DType.F32) return t.AsReadOnlySpan<float>().ToArray();
        ReadOnlySpan<ushort> bits = t.AsReadOnlySpan<ushort>();
        float[] values = new float[bits.Length];
        bool half = t.DType == DType.F16;
        for (int i = 0; i < bits.Length; i++)
            values[i] = half ? (float)BitConverter.UInt16BitsToHalf(bits[i]) : BitConverter.UInt32BitsToSingle((uint)bits[i] << 16);
        return values;
    }

    private static Tensor Load(TensorShape shape, float[] values, DType dt)
    {
        Tensor f32 = new(shape, DType.F32);
        values.CopyTo(f32.AsSpan<float>());
        if (dt == DType.F32) return f32;
        try { return f32.CastTo(dt); } finally { f32.Dispose(); }
    }
}

/// <summary>The contract reference (<see cref="AttentionHostReference"/>) on a case's checked rows, in the case's layout.</summary>
public static class AttnOracle
{
    /// <summary>Expected output rows <see cref="AttnCase.CheckedRows"/> of every (batch, head), flattened as the backend's output is.</summary>
    public static float[] Expected(AttnCase c, AttnInputs inputs)
    {
        (int first, int count) = c.CheckedRows;
        bool tm = c.Layout == AttnLayout.TokenMajor;
        TensorShape ks = tm ? new(c.Skv, c.KvHeads * c.HeadDim) : new(c.Batch, c.KvHeads, c.Skv, c.HeadDim);
        TensorShape qs = tm ? new(count, c.Heads * c.HeadDim) : new(c.Batch, c.Heads, count, c.HeadDim);
        using Tensor q = new(qs, DType.F32), k = Of(ks, inputs.K), v = Of(ks, inputs.V), o = new(qs, DType.F32);
        Span<float> qd = q.AsSpan<float>();
        int rowWidth = tm ? c.Heads * c.HeadDim : c.HeadDim;
        if (tm)
            inputs.Q.AsSpan(first * rowWidth, count * rowWidth).CopyTo(qd);
        else
            for (int bh = 0; bh < c.Batch * c.Heads; bh++)
                inputs.Q.AsSpan((bh * c.Sq + first) * c.HeadDim, count * c.HeadDim).CopyTo(qd.Slice(bh * count * c.HeadDim));
        using Tensor? mask = inputs.Mask is null ? null : SliceMaskRows(c, inputs.Mask, first, count);
        AttentionCall call = c.Flash
            ? AttentionCall.FromFlashAttention(o, q, k, v, c.KvLen, c.Causal, c.QOffset + first, c.Scale, 0f, null, c.Window, null)
            : tm ? AttentionCall.FromTokenMajor(o, q, k, v, mask, c.Heads, c.KvHeads, c.HeadDim, c.Scale, allowF16: false)
            : AttentionCall.FromScaledDotProductAttention(o, q, k, v, mask, c.Scale, allowF16: false);
        AttentionHostReference.Run(in call);
        return o.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>The same rows read out of a backend's full output.</summary>
    public static float[] CheckedRows(AttnCase c, float[] full)
    {
        (int first, int count) = c.CheckedRows;
        if (count == c.Sq) return full;
        bool tm = c.Layout == AttnLayout.TokenMajor;
        int rowWidth = tm ? c.Heads * c.HeadDim : c.HeadDim;
        float[] rows = new float[tm ? count * rowWidth : c.Batch * c.Heads * count * c.HeadDim];
        if (tm) full.AsSpan(first * rowWidth, count * rowWidth).CopyTo(rows);
        else
            for (int bh = 0; bh < c.Batch * c.Heads; bh++)
                full.AsSpan((bh * c.Sq + first) * c.HeadDim, count * c.HeadDim).CopyTo(rows.AsSpan(bh * count * c.HeadDim));
        return rows;
    }

    private static Tensor Of(TensorShape shape, float[] values)
    {
        Tensor t = new(shape, DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static Tensor SliceMaskRows(AttnCase c, float[] mask, int first, int count)
    {
        long[] s = c.MaskShape;
        long rows = s[^2], skv = s[^1];
        if (rows == 1 || count == rows) return Of(new TensorShape(s), mask);
        long blocks = mask.Length / (rows * skv);
        long[] shape = (long[])s.Clone();
        shape[^2] = count;
        float[] sliced = new float[blocks * count * skv];
        for (long blk = 0; blk < blocks; blk++)
            mask.AsSpan((int)((blk * rows + first) * skv), (int)(count * skv)).CopyTo(sliced.AsSpan((int)(blk * count * skv)));
        return Of(new TensorShape(shape), sliced);
    }
}

/// <summary>Agreement bars by achieved precision (inputs U[-1,1), scale 1/√D, oracle in double on dtype-rounded inputs).</summary>
public static class AttnTolerance
{
    /// <summary>(relative L2, minimum row cosine, maximum absolute difference) for <paramref name="achieved"/>.</summary>
    public static (double RelL2, double MinCos, double MaxAbs) For(AttnAchieved achieved) => achieved switch
    {
        AttnAchieved.F32 => (2e-5, 0.9999999, 1e-4),
        AttnAchieved.Tf32 => (2e-3, 0.99999, 5e-3),
        AttnAchieved.F16 => (3e-3, 0.99995, 8e-3),
        _ => (8e-3, 0.9999, 1.5e-2),
    };

    /// <summary>The worst precision <paramref name="kind"/>'s legacy entry may compute <paramref name="c"/> at.</summary>
    public static AttnAchieved Worst(string kind, AttnCase c)
    {
        if (c.DType == AttnDType.BF16) return AttnAchieved.Bf16Io;
        if (c.DType == AttnDType.F16) return AttnAchieved.F16;
        if (kind == "cpu") return AttnAchieved.F32;
        if (c.Flash)
            return kind == "cuda"
                ? (c.Sq > 1 ? AttnAchieved.F16 : AttnAchieved.F32)
                : (c.Causal && c.Sq >= 64 && c.HeadDim is 64 or 128 ? AttnAchieved.F16 : AttnAchieved.F32);
        if (c.AllowF16) return AttnAchieved.F16;
        return kind == "cuda" ? AttnAchieved.Tf32 : AttnAchieved.F32;
    }
}

/// <summary>Assertions over checked rows; xunit-free like <see cref="TensorAssert"/>.</summary>
public static class AttnAssert
{
    /// <summary>Throws unless <paramref name="actual"/> agrees with <paramref name="expected"/> within the bars; returns the measures.</summary>
    public static (double RelL2, double MinCos, double MaxAbs) Agrees(float[] actual, float[] expected, int rowWidth,
        (double RelL2, double MinCos, double MaxAbs) bar, string because)
    {
        (double rel, double cos, double max) = Measure(actual, expected, rowWidth);
        if (!(rel <= bar.RelL2 && cos >= bar.MinCos && max <= bar.MaxAbs))
            throw new InvalidOperationException(
                $"Attention disagrees with the oracle {because}: relL2 {rel:E2} (bar {bar.RelL2:E0}), min cos {cos:F7} "
                + $"(bar {bar.MinCos}), max |Δ| {max:E2} (bar {bar.MaxAbs:E0}).");
        return (rel, cos, max);
    }

    /// <summary>Throws unless the two differ by at least <paramref name="minRelL2"/>: a negative control's wrong oracle.</summary>
    public static double Differs(float[] actual, float[] wrong, int rowWidth, double minRelL2, string because)
    {
        double rel = Measure(actual, wrong, rowWidth).RelL2;
        if (!(rel >= minRelL2))
            throw new InvalidOperationException($"The output matches the wrong oracle {because}: relL2 {rel:E2} < {minRelL2}.");
        return rel;
    }

    /// <summary>Relative L2 over all elements, the minimum cosine over rows of <paramref name="rowWidth"/>, and the maximum
    /// absolute difference. A NaN anywhere makes every measure fail.</summary>
    public static (double RelL2, double MinCos, double MaxAbs) Measure(float[] actual, float[] expected, int rowWidth)
    {
        if (actual.Length != expected.Length)
            throw new InvalidOperationException($"Output has {actual.Length} elements; the oracle has {expected.Length}.");
        double num = 0, den = 0, max = 0, minCos = 1;
        for (int r = 0; r < actual.Length / rowWidth; r++)
        {
            double dot = 0, na = 0, ne = 0;
            for (int i = r * rowWidth; i < (r + 1) * rowWidth; i++)
            {
                double a = actual[i], e = expected[i], d = a - e;
                num += d * d;
                den += e * e;
                max = Math.Max(max, Math.Abs(d));
                if (double.IsNaN(d)) max = double.NaN;
                dot += a * e;
                na += a * a;
                ne += e * e;
            }
            double cos = na == 0 && ne == 0 ? 1 : dot / Math.Sqrt(na * ne);
            minCos = double.IsNaN(cos) ? double.NaN : Math.Min(minCos, cos);
        }
        return (Math.Sqrt(num / Math.Max(den, 1e-30)), minCos, max);
    }
}
