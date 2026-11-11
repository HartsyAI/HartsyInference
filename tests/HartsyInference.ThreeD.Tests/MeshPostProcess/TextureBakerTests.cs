using System.Numerics;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Unit lane, synthetic scenes: baking a known sparse volume onto a quad and checking individual texels.</summary>
public sealed class TextureBakerTests
{
    // 4x4x4 grid on [0,1]^3 (voxel 0.25). Layer z index 2 holds (x/3, y/3); layer 1 holds a constant 0.2 and layer 2's third channel is 0.8.
    private static SparseAttributeVolume ColorLayer()
    {
        List<int> coords = [];
        List<float> attrs = [];
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
            {
                coords.AddRange([x, y, 2]);
                attrs.AddRange([x / 3f, y / 3f]);
            }
        return new SparseAttributeVolume([.. coords], [.. attrs], 2, (4, 4, 4), Vector3.Zero, 0.25f);
    }

    private static Mesh Quad(float z)
    {
        return new Mesh
        {
            Vertices = [0, 0, z, 1, 0, z, 1, 1, z, 0, 1, z],
            Indices = [0, 1, 2, 0, 2, 3],
            Uvs = [0, 0, 1, 0, 1, 1, 0, 1],
        };
    }

    [Fact]
    public void Bake_ReproducesAKnownColorFieldTexelByTexel()
    {
        const int size = 8;

        BakedAttributes baked = TextureBaker.Bake(Quad(0.625f), null, ColorLayer(), size, size);

        Assert.All(baked.Covered, covered => Assert.True(covered));
        for (int ty = 0; ty < size; ty++)
            for (int tx = 0; tx < size; tx++)
            {
                // Texel center -> grid units (x4); voxel centers sit at index + 0.5, edges clamp by renormalization.
                double gx = 4.0 * (tx + 0.5) / size, gy = 4.0 * (ty + 0.5) / size;
                double expectedX = Math.Clamp(gx - 0.5, 0, 3) / 3.0, expectedY = Math.Clamp(gy - 0.5, 0, 3) / 3.0;
                int texel = ty * size + tx;
                Assert.Equal(expectedX, baked.Data[texel * 2], 4);
                Assert.Equal(expectedY, baked.Data[texel * 2 + 1], 4);
            }
    }

    [Fact]
    public void Bake_SnapsTexelPositionsToTheSourceSurfaceBeforeSampling()
    {
        // Two layers: z index 1 -> 0.2, z index 2 -> 0.8. The quad floats at z = 0.2 (inside layer 0, which is empty);
        // the source surface sits at z = 0.625, the center of layer 2.
        List<int> coords = [];
        List<float> attrs = [];
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
            {
                coords.AddRange([x, y, 1, x, y, 2]);
                attrs.AddRange([0.2f, 0.8f]);
            }
        SparseAttributeVolume volume = new([.. coords], [.. attrs], 1, (4, 4, 4), Vector3.Zero, 0.25f);
        Mesh floating = Quad(0.2f);
        TriangleBvh source = new(Quad(0.625f));

        BakedAttributes snapped = TextureBaker.Bake(floating, source, volume, 4, 4);
        BakedAttributes direct = TextureBaker.Bake(floating, null, volume, 4, 4);

        Assert.All(snapped.Data, v => Assert.Equal(0.8f, v, 4));
        Assert.All(direct.Data, v => Assert.Equal(0.2f, v, 4));
    }

    [Fact]
    public void Bake_LeavesTexelsOutsideTheUvCoverageUncovered()
    {
        Mesh half = new()
        {
            Vertices = [0, 0, 0.625f, 1, 0, 0.625f, 0, 1, 0.625f],
            Indices = [0, 1, 2],
            Uvs = [0, 0, 1, 0, 0, 1],
        };

        BakedAttributes baked = TextureBaker.Bake(half, null, ColorLayer(), 8, 8);

        Assert.True(baked.Covered[0]);
        Assert.False(baked.Covered[63]);
        Assert.Equal(0f, baked.Data[63 * 2]);
    }

    [Fact]
    public void Bake_TreatsTexelsWhoseVolumeNeighborsAreAllEmptyAsUncovered()
    {
        Mesh far = new()
        {
            Vertices = [0, 0, 0.05f, 1, 0, 0.05f, 1, 1, 0.05f, 0, 1, 0.05f],
            Indices = [0, 1, 2, 0, 2, 3],
            Uvs = [0, 0, 1, 0, 1, 1, 0, 1],
        };

        BakedAttributes baked = TextureBaker.Bake(far, null, ColorLayer(), 4, 4);

        Assert.All(baked.Covered, covered => Assert.False(covered));
    }

    [Fact]
    public void Bake_RequiresUvsAndAMatchingMesh()
    {
        Assert.Throws<ArgumentException>(() => TextureBaker.Bake(SyntheticMeshes.Grid(1), null, ColorLayer(), 4, 4));
    }
}
