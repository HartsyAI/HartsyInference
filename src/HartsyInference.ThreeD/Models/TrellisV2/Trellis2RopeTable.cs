using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// Upstream <c>RotaryPositionEmbedder</c> for 3D positions: each head carries <c>f = headDim / 6</c> rotation pairs per axis, in the
/// order x, y, z, at frequencies <c>10000^(-k/f)</c>. The pairs are adjacent channels <c>(2j, 2j + 1)</c>, and the
/// <c>headDim - 6f</c> trailing channels pass through (a zero phase). Cos and sin rows are <c>[1, tokens, 3f]</c>.
/// </summary>
internal sealed class Trellis2RopeTable : IDisposable
{
    private const double Theta = 10000.0;

    private readonly int _rotaryDim;
    private int _disposed;

    private Trellis2RopeTable(Tensor cos, Tensor sin, int rotaryDim)
    {
        Cos = cos;
        Sin = sin;
        _rotaryDim = rotaryDim;
    }

    internal Tensor Cos { get; }

    internal Tensor Sin { get; }

    internal int Tokens => (int)Cos.Shape[1];

    /// <summary>Rotation pairs per axis for a head width.</summary>
    internal static int PairsPerAxis(int headDim) => headDim / 6;

    /// <summary>Builds the table for <paramref name="tokens"/> positions given as <c>(x, y, z)</c> triples.</summary>
    internal static Trellis2RopeTable Build(int headDim, ReadOnlySpan<float> positions, int tokens)
    {
        int axisDim = 2 * PairsPerAxis(headDim);
        if (axisDim == 0)
            throw new ArgumentException($"Head width {headDim} is too small for a 3-axis RoPE.", nameof(headDim));
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [axisDim, axisDim, axisDim], Theta = Theta });
        Tensor cos = new(new TensorShape(1, tokens, plan.Pairs), DType.F32);
        Tensor sin = new(new TensorShape(1, tokens, plan.Pairs), DType.F32);
        try
        {
            MultiAxisRopeTables.ComputePairs(plan, positions, tokens, RopeAngleOrder.MultiplyByInverse, cos.AsSpan<float>(), sin.AsSpan<float>());
            return new Trellis2RopeTable(cos, sin, 3 * axisDim);
        }
        catch
        {
            cos.Dispose();
            sin.Dispose();
            throw;
        }
    }

    /// <summary>Positions <c>(x, y, z)</c> of the cells of a dense <paramref name="resolution"/>³ grid in x-major token order.</summary>
    internal static float[] DenseGridPositions(int resolution)
    {
        float[] positions = new float[resolution * resolution * resolution * 3];
        int i = 0;
        for (int x = 0; x < resolution; x++)
        {
            for (int y = 0; y < resolution; y++)
            {
                for (int z = 0; z < resolution; z++)
                {
                    positions[i++] = x;
                    positions[i++] = y;
                    positions[i++] = z;
                }
            }
        }

        return positions;
    }

    /// <summary>Positions <c>(x, y, z)</c> of voxels given as <c>(batch, x, y, z)</c> rows.</summary>
    internal static float[] VoxelPositions(ReadOnlySpan<int> coords)
    {
        int n = coords.Length / 4;
        float[] positions = new float[n * 3];
        for (int i = 0; i < n; i++)
        {
            positions[i * 3] = coords[i * 4 + 1];
            positions[i * 3 + 1] = coords[i * 4 + 2];
            positions[i * 3 + 2] = coords[i * 4 + 3];
        }

        return positions;
    }

    /// <summary>Rotates head-major <c>x [1, heads, tokens, headDim]</c> in place over the rotary channels.</summary>
    internal void Apply(IBackend backend, Tensor x)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        if (x.Shape.Rank != 4 || x.Shape[0] != 1 || x.Shape[2] != Tokens)
            throw new ArgumentException($"RoPE input must be [1, heads, {Tokens}, headDim]; got {x.Shape}.", nameof(x));
        StridedView vectors = RotaryViews.HeadMajor(x.Shape);
        StridedView table = RotaryViews.SharedTable(Cos.Shape, 1, x.Shape[1]);
        backend.Rotary(new RotaryCall(
            x, Cos, Sin, RotaryViews.Create(vectors, table, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPrefix, _rotaryDim)));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Cos.Dispose();
        Sin.Dispose();
    }
}
