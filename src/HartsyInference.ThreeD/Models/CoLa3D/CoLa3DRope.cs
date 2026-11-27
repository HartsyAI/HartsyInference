using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>3D RoPE for CoLa3D: each head's first <c>6·f</c> channels rotate over three axes with <c>f = headDim/6</c> pairs
/// per axis, the rest pass through. Built on the Core frequency plan and applied by the interleaved RoPE op.</summary>
internal static class CoLa3DRope
{
    private const double Theta = 10000.0;

    internal static int PairsPerAxis(int headDim) => headDim / 6;

    internal static int RotaryDim(int headDim) => 6 * PairsPerAxis(headDim);

    internal static RopeFrequencyPlan Plan(int headDim)
    {
        int axis = 2 * PairsPerAxis(headDim);
        return RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [axis, axis, axis], Theta = Theta });
    }

    /// <summary>Angle tables for <paramref name="tokens"/> positions given as <c>(x, y, z)</c> triples.</summary>
    internal static CoLa3DRopeTable Build(RopeFrequencyPlan plan, ReadOnlySpan<float> positions, int tokens)
    {
        CoLa3DRopeTable table = new(tokens, plan.Pairs);
        MultiAxisRopeTables.ComputePairs(
            plan, positions, tokens, RopeAngleOrder.MultiplyByInverse, table.Cos.AsSpan<float>(), table.Sin.AsSpan<float>());
        return table;
    }

    /// <summary>Rotates <c>x [1, tokens, heads, headDim]</c> in place over the rotary channels.</summary>
    internal static void Apply(IBackend backend, Tensor x, CoLa3DRopeTable table, int rotaryDim)
    {
        // Tokens share one [L, pairs] table across heads; the rotation runs over the token-major vectors.
        StridedView vectors = RotaryViews.TokenMajor(x.Shape);
        StridedView tableView = RotaryViews.SharedTable(table.Cos.Shape, 1, x.Shape[2]);
        backend.Rotary(new RotaryCall(x, table.Cos, table.Sin, RotaryViews.Create(vectors, tableView, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPrefix, rotaryDim)));
    }
}

/// <summary>Owned cos/sin rows <c>[1, tokens, pairs]</c> for one set of positions.</summary>
internal sealed class CoLa3DRopeTable : IDisposable
{
    private int _disposed;

    internal CoLa3DRopeTable(int tokens, int pairs)
    {
        Cos = CoLa3DOps.Alloc(1, tokens, pairs);
        Sin = CoLa3DOps.Alloc(1, tokens, pairs);
    }

    internal Tensor Cos { get; }

    internal Tensor Sin { get; }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Cos.Dispose();
        Sin.Dispose();
    }
}
