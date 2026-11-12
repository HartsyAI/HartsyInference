using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Trellis;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamGaussianRotationTests
{
    private const float C45 = 0.70710678f;

    // One voxel whose first Gaussian has raw rotation channels (0, 1, 0, 0), the rest zero.
    private static GaussianSplatCloud Build(float? rotationScale)
    {
        using Tensor feats = new(new TensorShape(1, 448), DType.F32);
        feats.AsSpan<float>()[288 + 1] = 1f;
        SparseTensor netOut = new(feats, [0, 3, 4, 5], 64);
        return rotationScale is { } scale ? TrellisGaussianRepresentation.Build(netOut, scale) : TrellisGaussianRepresentation.Build(netOut);
    }

    [Fact]
    public void Build_MultipliesTheRawRotationByTheDecoderConfigsLrBeforeAddingTheIdentity()
    {
        GaussianSplatCloud cloud = Build(0.1f);

        // Upstream: normalize(0.1 * raw + (1, 0, 0, 0)) = (1, 0.1, 0, 0) before the y-up rotation q_R = (cos 45, sin 45, 0, 0).
        Assert.Equal(C45 * 0.9f, cloud.Rotations[0], 5);
        Assert.Equal(C45 * 1.1f, cloud.Rotations[1], 5);
        Assert.Equal(0f, cloud.Rotations[2], 5);
        Assert.Equal(0f, cloud.Rotations[3], 5);
    }

    [Fact]
    public void Build_WithoutAScaleKeepsTheRawChannelsAsTheTrellisPipelineAlwaysDid()
    {
        GaussianSplatCloud cloud = Build(null);

        Assert.Equal(C45 * 0f, cloud.Rotations[0], 5);   // qw 1 - qx 1
        Assert.Equal(C45 * 2f, cloud.Rotations[1], 5);   // qx 1 + qw 1
    }
}
