using System.Numerics;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Io;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Unit lane, synthetic sphere and shell volume: the whole chain from raw mesh to GLB.</summary>
public sealed class PbrMeshPostProcessorTests
{
    private const int Grid = 32;
    private const float Radius = 0.5f;

    // Shell of voxels around a sphere of radius 0.5 in [-0.5, 0.5]^3; base color encodes position, metallic .25, roughness .75, alpha 1.
    private static SparseAttributeVolume ShellVolume()
    {
        float voxel = 1f / Grid;
        List<int> coords = [];
        List<float> attrs = [];
        for (int z = 0; z < Grid; z++)
            for (int y = 0; y < Grid; y++)
                for (int x = 0; x < Grid; x++)
                {
                    float px = -0.5f + (x + 0.5f) * voxel, py = -0.5f + (y + 0.5f) * voxel, pz = -0.5f + (z + 0.5f) * voxel;
                    float r = MathF.Sqrt(px * px + py * py + pz * pz);
                    if (MathF.Abs(r - Radius) > 2.5f * voxel) continue;
                    coords.AddRange([x, y, z]);
                    attrs.AddRange([px + 0.5f, py + 0.5f, pz + 0.5f, 0.25f, 0.75f, 1f]);
                }
        return new SparseAttributeVolume([.. coords], [.. attrs], 6, (Grid, Grid, Grid), new Vector3(-0.5f), voxel);
    }

    private static PbrPostProcessOptions SmallOptions() => new() { DecimationTarget = 300, TextureSize = 128, PaddingTexels = 3 };

    [Fact]
    public void Process_ProducesAnUnwrappedDecimatedMeshAndBakedMaterial()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(4, Radius);

        PbrPostProcessResult result = PbrMeshPostProcessor.Process(sphere, ShellVolume(), PbrAttributeLayout.Default, SmallOptions());

        Assert.InRange(result.Mesh.TriangleCount, 250, 340);
        Assert.NotNull(result.Mesh.Uvs);
        Assert.NotNull(result.Mesh.Normals);
        foreach (float uv in result.Mesh.Uvs!) Assert.InRange(uv, 0f, 1f);
        Assert.Equal((128, 128, 3), (result.Material.BaseColor.Width, result.Material.BaseColor.Height, result.Material.BaseColor.Channels));
        Assert.NotNull(result.Material.Alpha);
        Assert.InRange(result.CoveredTexels, 2000, 128 * 128);
    }

    [Fact]
    public void Process_ClosesAPinholeWithinTheOpeningWidthAndLeavesItWithoutOne()
    {
        // Two neighbouring faces removed: a quad hole whose zip edge is longer than the perimeter fill's reach but within the opening width.
        Mesh sphere = SyntheticMeshes.Icosphere(4, Radius);
        int[] indices = sphere.Indices[6..];
        Mesh holed = new() { Vertices = sphere.Vertices, Indices = indices };
        PbrPostProcessOptions open = new() { DecimationTarget = 100_000, TextureSize = 128, PaddingTexels = 3, HoleFillMaxPerimeter = 1e-3, OpeningWidthRatio = 0 };

        long without = MeshStatistics.Measure(PbrMeshPostProcessor.Process(holed, ShellVolume(), PbrAttributeLayout.Default, open).Mesh).BoundaryEdges;
        long with = MeshStatistics.Measure(PbrMeshPostProcessor.Process(holed, ShellVolume(), PbrAttributeLayout.Default, open with { OpeningWidthRatio = 0.1 }).Mesh).BoundaryEdges;

        Assert.True(without > 0);
        Assert.Equal(0, with);
    }

    [Fact]
    public void Process_RejectsANegativeOpeningWidth()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2, Radius);

        Assert.Throws<ArgumentOutOfRangeException>(() => PbrMeshPostProcessor.Process(
            sphere, ShellVolume(), PbrAttributeLayout.Default, SmallOptions() with { OpeningWidthRatio = -0.1 }));
    }

    [Fact]
    public void Process_AcceptsAZeroPaddingRequest()
    {
        // PaddingTexels = 0 was a valid request before the unwrapper had a minimum gutter; it must keep working through the PBR options.
        Mesh sphere = SyntheticMeshes.Icosphere(3, Radius);

        PbrPostProcessResult result = PbrMeshPostProcessor.Process(
            sphere, ShellVolume(), PbrAttributeLayout.Default, new PbrPostProcessOptions { DecimationTarget = 300, TextureSize = 128, PaddingTexels = 0 });

        Assert.True(result.CoveredTexels > 0);
        foreach (float uv in result.Mesh.Uvs!) Assert.InRange(uv, 0f, 1f);
    }

    [Fact]
    public void Process_BakesTheVolumeColorAtEveryCoveredTexel()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(4, Radius);
        PbrPostProcessResult result = PbrMeshPostProcessor.Process(sphere, ShellVolume(), PbrAttributeLayout.Default, SmallOptions());
        RasterizedUvMap raster = Rasterizer.Rasterize(result.Mesh.Uvs!, result.Mesh.Indices, 128, 128);

        int checkedTexels = 0;
        for (int texel = 0; texel < 128 * 128; texel += 7)
        {
            int tri = raster.TriangleIds[texel];
            if (tri < 0) continue;
            checkedTexels++;
            float b1 = raster.Barycentric[texel * 2], b2 = raster.Barycentric[texel * 2 + 1], b0 = 1 - b1 - b2;
            float[] p = new float[3];
            for (int a = 0; a < 3; a++)
                p[a] = b0 * result.Mesh.Vertices[result.Mesh.Indices[tri * 3] * 3 + a]
                    + b1 * result.Mesh.Vertices[result.Mesh.Indices[tri * 3 + 1] * 3 + a]
                    + b2 * result.Mesh.Vertices[result.Mesh.Indices[tri * 3 + 2] * 3 + a];
            // The bake snaps onto the original sphere, so the color is the (p + 0.5) of a point near the decimated surface.
            for (int a = 0; a < 3; a++)
                Assert.InRange(result.Material.BaseColor.Pixels[texel * 3 + a] / 255.0, p[a] + 0.5 - 0.1, p[a] + 0.5 + 0.1);
            Assert.InRange((int)result.Material.Metallic.Pixels[texel], 63, 65);
            Assert.InRange((int)result.Material.Roughness.Pixels[texel], 190, 193);
            Assert.Equal(255, result.Material.Alpha!.Pixels[texel]);
        }
        Assert.True(checkedTexels > 200);
    }

    // A high-contrast colour pattern (voxels alternate between 0.15 and 0.85 per channel) so the covered texels have a sharp structure at every chart border.
    private static SparseAttributeVolume CheckerVolume()
    {
        float voxel = 1f / Grid;
        List<int> coords = [];
        List<float> attrs = [];
        for (int z = 0; z < Grid; z++)
            for (int y = 0; y < Grid; y++)
                for (int x = 0; x < Grid; x++)
                {
                    float px = -0.5f + (x + 0.5f) * voxel, py = -0.5f + (y + 0.5f) * voxel, pz = -0.5f + (z + 0.5f) * voxel;
                    if (MathF.Abs(MathF.Sqrt(px * px + py * py + pz * pz) - Radius) > 2.5f * voxel) continue;
                    coords.AddRange([x, y, z]);
                    float a = (x + y + z) % 2 == 0 ? 0.15f : 0.85f, b = (x + 2 * y) % 2 == 0 ? 0.85f : 0.15f, c = (y + z) % 2 == 0 ? 0.15f : 0.85f;
                    attrs.AddRange([a, b, c, 0.25f, 0.75f, 1f]);
                }
        return new SparseAttributeVolume([.. coords], [.. attrs], 6, (Grid, Grid, Grid), new Vector3(-0.5f), voxel);
    }

    [Fact]
    public void Process_GutterColoursStayInsideTheRangeOfTheBakedTexels()
    {
        // The inpaint is bounded by its neighbours: a gutter texel never leaves the per-channel range of the covered texels (without the bound the gradient correction overshoots at the sharp chart borders).
        Mesh sphere = SyntheticMeshes.Icosphere(2, Radius);
        PbrPostProcessOptions options = new() { DecimationTarget = 80, TextureSize = 128, PaddingTexels = 3 };

        PbrPostProcessResult result = PbrMeshPostProcessor.Process(sphere, CheckerVolume(), PbrAttributeLayout.Default, options);
        RasterizedUvMap raster = Rasterizer.Rasterize(result.Mesh.Uvs!, result.Mesh.Indices, 128, 128);

        int[] low = [255, 255, 255], high = [0, 0, 0];
        for (int texel = 0; texel < 128 * 128; texel++)
            if (raster.TriangleIds[texel] >= 0)
                for (int c = 0; c < 3; c++)
                {
                    int v = result.Material.BaseColor.Pixels[texel * 3 + c];
                    low[c] = Math.Min(low[c], v);
                    high[c] = Math.Max(high[c], v);
                }

        int gutterTexels = 0;
        for (int texel = 0; texel < 128 * 128; texel++)
        {
            if (raster.TriangleIds[texel] >= 0) continue;
            gutterTexels++;
            for (int c = 0; c < 3; c++)
                Assert.InRange((int)result.Material.BaseColor.Pixels[texel * 3 + c], low[c] - 1, high[c] + 1);
        }

        Assert.True(gutterTexels > 1000, "the test needs a gutter to check");
    }

    [Fact]
    public void Process_FillsGuttersSoNoTexelStaysBlack()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(3, Radius);

        PbrPostProcessResult result = PbrMeshPostProcessor.Process(sphere, ShellVolume(), PbrAttributeLayout.Default, SmallOptions());

        // Alpha is 1 everywhere in the volume, so after inpainting every texel with a known neighbor must be 255.
        Assert.All(result.Material.Alpha!.Pixels, a => Assert.InRange((int)a, 250, 255));
    }

    [Fact]
    public void ProcessToGlb_IsDeterministicAndAValidGlb()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(3, Radius);
        PbrGlbOptions glb = new() { ConvertZUpToYUp = true, DoubleSided = true };

        byte[] a = PbrMeshPostProcessor.ProcessToGlb(sphere, ShellVolume(), PbrAttributeLayout.Default, SmallOptions(), glb);
        byte[] b = PbrMeshPostProcessor.ProcessToGlb(sphere, ShellVolume(), PbrAttributeLayout.Default, SmallOptions(), glb);

        Assert.Equal(a, b);
        Mesh read = GlbReader.Parse(a);
        Assert.True(read.TriangleCount > 0);
    }

    [Fact]
    public void Process_DoesNotModifyItsInputs()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(3, Radius);
        float[] vertices = (float[])sphere.Vertices.Clone();
        int[] indices = (int[])sphere.Indices.Clone();

        PbrMeshPostProcessor.Process(sphere, ShellVolume(), PbrAttributeLayout.Default, SmallOptions());

        Assert.Equal(vertices, sphere.Vertices);
        Assert.Equal(indices, sphere.Indices);
        Assert.Null(sphere.Normals);
    }

    [Fact]
    public void Process_HonoursCancellationAndValidatesTheLayout()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2, Radius);
        using CancellationTokenSource cts = new();
        cts.Cancel();

        Assert.Throws<OperationCanceledException>(
            () => PbrMeshPostProcessor.Process(sphere, ShellVolume(), PbrAttributeLayout.Default, SmallOptions(), cts.Token));
        Assert.Throws<ArgumentException>(
            () => PbrMeshPostProcessor.Process(sphere, ShellVolume(), new PbrAttributeLayout(0, 3, 4, 9), SmallOptions()));
    }
}
