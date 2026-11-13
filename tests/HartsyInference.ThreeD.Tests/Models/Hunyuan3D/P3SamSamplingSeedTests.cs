using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Hunyuan3D.Part;
using HartsyInference.ThreeD.Tests.MeshPostProcess;

namespace HartsyInference.ThreeD.Tests.Models.Hunyuan3D;

/// <summary>P3-SAM's surface points depend on the seed and on nothing else: the same seed gives the same points and faces, another seed gives others. A 3D request without a seed draws a random one before it runs
/// (<c>MeshService.GenerateAsync</c>), which is why three CLI runs on one bust gave three different splits; with <c>--seed</c> the split is repeatable.</summary>
public sealed class P3SamSamplingSeedTests
{
    [Fact]
    public void TheSameSeedGivesTheSamePointsNormalsAndFaces_AnotherSeedDoesNot()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(3);

        (double[] points, float[] normals, int[] face) = P3SamAutoMask.Sample(sphere.Vertices, sphere.Indices, 2_000, new Random(42));
        (double[] samePoints, float[] sameNormals, int[] sameFace) = P3SamAutoMask.Sample(sphere.Vertices, sphere.Indices, 2_000, new Random(42));
        (double[] otherPoints, _, int[] otherFace) = P3SamAutoMask.Sample(sphere.Vertices, sphere.Indices, 2_000, new Random(43));

        Assert.Equal(points, samePoints);
        Assert.Equal(normals, sameNormals);
        Assert.Equal(face, sameFace);
        Assert.NotEqual(points, otherPoints);
        Assert.NotEqual(face, otherFace);
    }
}
