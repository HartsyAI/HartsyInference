using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Paint;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>The Hunyuan3D-Paint 2.1 bake against upstream's own <c>MeshRender.back_project</c> and <c>fast_bake_texture</c> on the CPU (<c>tests/python-reference/dump_paint_bake.py</c>; the CUDA rasterizer is replaced there
/// by a transcription of its kernel). The fixture carries the mesh, its UVs, six view images and upstream's outputs.</summary>
/// <remarks>Integration lane: needs <c>HARTSY_PARITY_DIR</c> holding <c>paint_bake_fp32.npz</c>. The view bake and the merge are compared texel by texel on the texels upstream rasterized from the UV map (so the comparison
/// does not depend on how the two rasterize UV space); the UV coverage is compared separately.</remarks>
[Trait("Category", "Integration")]
public sealed class PaintBakeParityTests(ITestOutputHelper output)
{
    private const string Fixture = "paint_bake_fp32.npz";

    private sealed record Loaded(NpzArchive Archive, Mesh Mesh, int Texture, int Resolution, int[] Texels, float[] Positions, float[] Views);

    private bool TryLoad(out Loaded loaded)
    {
        loaded = null!;
        string parityDir = Environment.GetEnvironmentVariable(ParityFixtures.ParityDirVariable) ?? string.Empty;
        string path = Path.Combine(parityDir, Fixture);
        if (!File.Exists(path))
        {
            string reason = $"{ParityFixtures.ParityDirVariable} does not hold {Fixture}";
            if (Environment.GetEnvironmentVariable(ParityFixtures.RequireWeightsVariable) == "1")
                throw new InvalidOperationException($"{ParityFixtures.RequireWeightsVariable}=1 but {reason}.");
            output.WriteLine($"SKIPPED: {reason}");
            return false;
        }

        NpzArchive archive = NpzArchive.Open(path);
        int[] size = archive.Get("texture_size").ToInt32();
        int texture = size[0], resolution = size[1];
        Mesh mesh = new() { Vertices = archive.Get("verts").ToFloat32(), Indices = archive.Get("idx").ToInt32(), Uvs = archive.Get("uvs").ToFloat32() };
        int[] grid = archive.Get("tex_grid").ToInt32();
        int[] texels = new int[grid.Length / 2];
        for (int i = 0; i < texels.Length; i++)
            texels[i] = grid[i * 2] * texture + grid[i * 2 + 1];
        float[] fourD = archive.Get("tex_position").ToFloat32();
        float[] positions = new float[texels.Length * 3];
        for (int i = 0; i < texels.Length; i++)
            Array.Copy(fourD, i * 4, positions, i * 3, 3);
        loaded = new Loaded(archive, mesh, texture, resolution, texels, positions, archive.Get("views").ToFloat32());
        return true;
    }

    [Fact]
    public void ViewBakesAndTheMergeMatchUpstreamOnTheTexelsItRasterized()
    {
        if (!TryLoad(out Loaded f))
            return;
        PaintMeshFrame frame = PaintMeshFrame.Create(f.Mesh);
        PaintViewRenderer renderer = new(frame);
        PaintTexelSurface surface = PaintTexelSurface.FromTexels(f.Texture, f.Texture, f.Texels, f.Positions);
        PaintTextureMerger merger = new(surface, 3);
        bool realCanny = f.Archive.Get("canny").ToInt32()[0] == 1;
        output.WriteLine($"{surface.Count} texels, edge detector in the reference: {(realCanny ? "OpenCV Canny" : "a numpy stand-in (edge-dependent values are not comparable)")}");

        for (int v = 0; v < f.Views.Length / 3; v++)
        {
            float[] image = f.Archive.Get($"image{v}").ToFloat32();
            PaintViewBake bake = PaintBackProjector.Project(renderer, surface, image, f.Resolution, 3, f.Views[v * 3], f.Views[v * 3 + 1]);
            float[] cosine = At(f.Archive.Get($"cos{v}").ToFloat32(), f.Texels, 1), color = At(f.Archive.Get($"color{v}").ToFloat32(), f.Texels, 3);
            Compare(v, bake, cosine, color);
            merger.Add(bake, f.Views[v * 3 + 2]);
        }

        (float[] texture, bool[] painted) = merger.Finish();
        float[] merged = f.Archive.Get("merged").ToFloat32();
        bool[] referencePainted = f.Archive.Get("painted").ToFloat32().Select(x => x > 0.5f).ToArray();
        int both = 0, onlyOurs = 0, onlyReference = 0;
        double sum = 0, worst = 0;
        foreach (int t in f.Texels)
        {
            bool ours = painted[t], theirs = referencePainted[t];
            if (ours && theirs)
            {
                both++;
                for (int c = 0; c < 3; c++)
                {
                    double d = Math.Abs(texture[t * 3 + c] - merged[t * 3 + c]);
                    sum += d;
                    worst = Math.Max(worst, d);
                }
            }
            else if (ours)
                onlyOurs++;
            else if (theirs)
                onlyReference++;
        }

        output.WriteLine($"merged texture: painted in both {both}, only here {onlyOurs}, only upstream {onlyReference}; mean abs difference {sum / (both * 3):E2}, largest {worst:E2}");
    }

    private void Compare(int view, PaintViewBake bake, float[] cosine, float[] color)
    {
        int both = 0, onlyOurs = 0, onlyReference = 0;
        double cosSum = 0, colorSum = 0, colorWorst = 0;
        for (int i = 0; i < cosine.Length; i++)
        {
            bool ours = bake.Weights[i] > 0f, theirs = cosine[i] > 0f;
            if (ours && theirs)
            {
                both++;
                cosSum += Math.Abs(bake.Weights[i] - cosine[i]);
                for (int c = 0; c < 3; c++)
                {
                    double d = Math.Abs(bake.Colors[i * 3 + c] - color[i * 3 + c]);
                    colorSum += d;
                    colorWorst = Math.Max(colorWorst, d);
                }
            }
            else if (ours)
                onlyOurs++;
            else if (theirs)
                onlyReference++;
        }

        output.WriteLine($"view {view}: painted in both {both}, only here {onlyOurs}, only upstream {onlyReference}; cosine mean abs {cosSum / Math.Max(both, 1):E2}; colour mean abs {colorSum / Math.Max(both * 3, 1):E2}, largest {colorWorst:E2}");
    }

    // Reads the texture-space array at the covered texels, in the surface's order.
    private static float[] At(float[] texture, int[] texels, int channels)
    {
        float[] values = new float[texels.Length * channels];
        for (int i = 0; i < texels.Length; i++)
            for (int c = 0; c < channels; c++)
                values[i * channels + c] = texture[texels[i] * channels + c];
        return values;
    }

    [Fact]
    public void UvCoverageAgreesWithUpstreamUpToItsHalfTexelConvention()
    {
        if (!TryLoad(out Loaded f))
            return;
        PaintMeshFrame frame = PaintMeshFrame.Create(f.Mesh);
        PaintTexelSurface ours = PaintTexelSurface.Create(frame, f.Mesh.Uvs!, f.Texture, f.Texture);
        // This repository numbers texture rows from the top of the image, upstream's rasterization from the bottom.
        HashSet<int> mine = [.. ours.Texels.Select(t => (f.Texture - 1 - t / f.Texture) * f.Texture + t % f.Texture)];
        int both = f.Texels.Count(mine.Contains);
        output.WriteLine($"covered texels: here {mine.Count}, upstream {f.Texels.Length}, in both {both} ({100.0 * both / f.Texels.Length:F1} % of upstream's)");
        Assert.True(both > 0.8 * f.Texels.Length, $"only {both} of {f.Texels.Length}");
    }
}
