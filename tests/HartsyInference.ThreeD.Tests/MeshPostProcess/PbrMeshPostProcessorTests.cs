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
