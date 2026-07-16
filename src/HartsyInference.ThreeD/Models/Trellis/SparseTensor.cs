using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>A TRELLIS sparse-voxel tensor: per-voxel features <see cref="Feats"/> <c>[N, C]</c> paired 1:1 with
/// integer voxel coordinates <see cref="Coords"/> <c>[N, 4]</c> = <c>(batch, x, y, z)</c> (host-side, ≤1023). Most ops
/// transform <see cref="Feats"/> only (via <see cref="Replace"/>, reusing coords); conv/down/up change the voxel set.
/// Inference is B=1 (single object), so full attention = dense attention over all N voxels. See
/// <c>docs/Research/TRELLIS_ARCHITECTURE.md</c> §sparse-subsystem.</summary>
public sealed class SparseTensor
{
    /// <summary>Per-voxel features <c>[N, C]</c>.</summary>
    public Tensor Feats { get; }

    /// <summary>Voxel coords, host int <c>[N·4]</c> laid <c>(b,x,y,z)</c> per voxel.</summary>
    public int[] Coords { get; }

    /// <summary>Grid resolution at this tensor's scale (64 at input, 32 after one downsample) — the extent for the
    /// dense scatter/gather used by submanifold conv.</summary>
    public int Resolution { get; }

    public int Count => Coords.Length / 4;
    public int Channels => (int)Feats.Shape[Feats.Shape.Rank - 1];

    public SparseTensor(Tensor feats, int[] coords, int resolution)
    {
        Feats = feats; Coords = coords; Resolution = resolution;
    }

    /// <summary>New tensor with the same coords/resolution and swapped-in features (the most-used sparse primitive).</summary>
    public SparseTensor Replace(Tensor feats) => new(feats, Coords, Resolution);
}
