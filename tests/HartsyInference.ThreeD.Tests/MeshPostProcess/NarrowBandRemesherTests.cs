using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Unit lane, synthetic meshes: the dual-contoured offset shell is closed, sits where the offset surface is, and closes what is narrower than the band.</summary>
public sealed class NarrowBandRemesherTests
{
    private const int Res = 64;

    private static NarrowBandRemeshOptions Options(bool outerOnly = true, double projectBack = 0) =>
        new() { Resolution = Res, OuterShellOnly = outerOnly, ProjectBack = projectBack };

    [Fact]
    public void ASphere_BecomesAClosedManifoldShellAtTheOffsetRadius()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(4, 1f);

        Mesh shell = NarrowBandRemesher.Remesh(sphere, Options());

        (double[] min, double[] max) = SyntheticMeshes.Bounds(shell);
        double voxel = (2.0 + 3.0 * 1 / Res * 2.0) / Res;
        Assert.True(SyntheticMeshes.IsClosedManifold(shell));
        Assert.Equal(2, SyntheticMeshes.EulerCharacteristic(shell));
        Assert.InRange(max[0], 1 + 0.5 * voxel, 1 + 2.5 * voxel);
        Assert.InRange(min[0], -(1 + 2.5 * voxel), -(1 + 0.5 * voxel));
        Assert.True(SyntheticMeshes.SignedVolume(shell) > 4.0 / 3 * Math.PI * 0.99);
    }

    [Fact]
    public void WithBothSheets_TheInnerOneIsTheReversedOffsetInwards()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(4, 1f);

        Mesh both = NarrowBandRemesher.Remesh(sphere, Options(outerOnly: false));
        Mesh outer = NarrowBandRemesher.Remesh(sphere, Options(outerOnly: true));

        Assert.InRange(both.TriangleCount, outer.TriangleCount * 18 / 10, outer.TriangleCount * 22 / 10);
        Assert.True(SyntheticMeshes.SignedVolume(both) < SyntheticMeshes.SignedVolume(outer));
    }

    [Fact]
    public void ATorus_KeepsItsGenus()
    {
        Mesh torus = SyntheticMeshes.Torus(48, 24, 1f, 0.4f);

        Mesh shell = NarrowBandRemesher.Remesh(torus, Options());

        Assert.True(SyntheticMeshes.IsClosedManifold(shell));
        Assert.Equal(0, SyntheticMeshes.EulerCharacteristic(shell));
    }

    [Fact]
    public void AHoleNarrowerThanTheBand_IsClosed()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(4, 1f);
        Mesh holed = new() { Vertices = sphere.Vertices, Indices = sphere.Indices[6..] };
        Assert.False(SyntheticMeshes.IsClosedManifold(holed));

        Mesh shell = NarrowBandRemesher.Remesh(holed, Options());

        Assert.True(SyntheticMeshes.IsClosedManifold(shell));
    }

    [Fact]
    public void ProjectingBack_MovesTheShellOntoTheInput()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(4, 1f);
        TriangleBvh bvh = new(sphere);

        double Distance(Mesh m) => Enumerable.Range(0, m.VertexCount).Average(i => bvh.Distance(m.Vertices[i * 3], m.Vertices[i * 3 + 1], m.Vertices[i * 3 + 2]));
        double offset = Distance(NarrowBandRemesher.Remesh(sphere, Options()));
        double projected = Distance(NarrowBandRemesher.Remesh(sphere, Options(projectBack: 0.9)));

        Assert.True(projected < offset * 0.25, $"offset {offset}, projected {projected}");
    }

    [Fact]
    public void TheSameInputGivesTheSameShell()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(3, 1f);

        Mesh a = NarrowBandRemesher.Remesh(sphere, Options());
        Mesh b = NarrowBandRemesher.Remesh(sphere, Options());

        Assert.Equal(a.Indices, b.Indices);
        Assert.Equal(a.Vertices, b.Vertices);
    }

    [Fact]
    public void TwoSeparateSpheres_EachKeepOnlyTheirOuterSheet()
    {
        Mesh a = SyntheticMeshes.Icosphere(3, 1f);
        Mesh b = SyntheticMeshes.Icosphere(3, 1f);
        float[] moved = [.. b.Vertices];
        for (int i = 0; i < moved.Length; i += 3)
            moved[i] += 3f;
        Mesh two = new() { Vertices = [.. a.Vertices, .. moved], Indices = [.. a.Indices, .. b.Indices.Select(i => i + a.VertexCount)] };

        Mesh outer = NarrowBandRemesher.Remesh(two, Options());
        Mesh both = NarrowBandRemesher.Remesh(two, Options(outerOnly: false));

        Assert.Equal(4, SyntheticMeshes.EulerCharacteristic(outer));
        Assert.InRange(outer.TriangleCount, both.TriangleCount * 4 / 10, both.TriangleCount * 6 / 10);
        Assert.True(SyntheticMeshes.SignedVolume(outer) > 2 * 4.0 / 3 * Math.PI * 0.95);
    }

    [Theory]
    [InlineData(1.5)]
    [InlineData(2.5)]
    public void AnotherBand_StillGivesAClosedShellAtTheMatchingOffset(double band)
    {
        Mesh sphere = SyntheticMeshes.Icosphere(4, 1f);

        Mesh shell = NarrowBandRemesher.Remesh(sphere, new NarrowBandRemeshOptions { Resolution = Res, Band = band, OuterShellOnly = true });

        (double[] min, double[] max) = SyntheticMeshes.Bounds(shell);
        double voxel = (2.0 * (Res + 3 * band) / Res) / Res;
        Assert.True(SyntheticMeshes.IsClosedManifold(shell));
        Assert.InRange(max[0], 1 + (band - 1.0) * voxel, 1 + (band + 1.0) * voxel);
        Assert.InRange(min[0], -(1 + (band + 1.0) * voxel), -(1 + (band - 1.0) * voxel));
    }

    [Fact]
    public void AResolutionThatHalvesToABaseGridIsAccepted()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2, 1f);

        Mesh shell = NarrowBandRemesher.Remesh(sphere, new NarrowBandRemeshOptions { Resolution = 100, OuterShellOnly = true });

        Assert.True(shell.TriangleCount > 0);
    }

    [Theory]
    [InlineData(31)]
    [InlineData(1001)]
    [InlineData(5000)]
    public void AResolutionThatIsNotAGridIsRefused(int resolution)
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2, 1f);

        Assert.Throws<ArgumentException>(() => NarrowBandRemesher.Remesh(sphere, new NarrowBandRemeshOptions { Resolution = resolution }));
    }
}
