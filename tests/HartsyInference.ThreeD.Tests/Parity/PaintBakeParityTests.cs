using System.Text.Json;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Paint;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>The Hunyuan3D-Paint 2.1 bake against upstream's own <c>MeshRender.back_project</c> and <c>fast_bake_texture</c> on the CPU (<c>tests/python-reference/dump_paint_bake.py</c>; the CUDA rasterizer is replaced there
/// by a transcription of its kernel). The fixture carries the mesh, its UVs, six view images and upstream's outputs.</summary>
/// <remarks>Integration lane: needs <c>HARTSY_PARITY_DIR</c> holding <c>paint_bake_fp32.npz</c>. The view bake and the merge are compared texel by texel on the texels upstream rasterized from the UV map (so the comparison
/// does not depend on how the two rasterize UV space); the UV coverage is compared separately. <see cref="WriteInputsForTheOracle"/> writes the inputs the oracle reads.</remarks>
[Trait("Category", "Integration")]
public sealed class PaintBakeParityTests(ITestOutputHelper output)
{
    private const string Fixture = "paint_bake_fp32.npz";
    private const string InputVariable = "HARTSY_PAINT_BAKE_INPUT_DIR", MeshVariable = "HARTSY_PAINT_BAKE_MESH", ReferenceVariable = "HARTSY_PAINT_BAKE_REF";

    private sealed record Loaded(NpzArchive Archive, Mesh Mesh, int Texture, int Resolution, int[] Texels, float[] Positions, float[] Views);

    /// <summary>Writes what <c>dump_paint_bake.py --in</c> reads: the mesh of <c>HARTSY_PAINT_BAKE_MESH</c> (an OBJ) welded and unwrapped as the pipeline does it for a texture of 1024, six views
    /// of 512 px made from the raw RGB image <c>HARTSY_PAINT_BAKE_REF</c> (square, 8-bit, rolled sideways by 37 px and darkened differently per view), into <c>HARTSY_PAINT_BAKE_INPUT_DIR</c>.</summary>
    [Fact]
    public void WriteInputsForTheOracle()
    {
        string? directory = Environment.GetEnvironmentVariable(InputVariable), meshPath = Environment.GetEnvironmentVariable(MeshVariable), referencePath = Environment.GetEnvironmentVariable(ReferenceVariable);
        if (string.IsNullOrEmpty(directory) || !File.Exists(meshPath) || !File.Exists(referencePath))
        {
            output.WriteLine($"SKIPPED: {InputVariable}, {MeshVariable} and {ReferenceVariable} must name a directory, an OBJ file and a raw RGB image.");
            return;
        }

        const int texture = 1024, resolution = 512;
        Directory.CreateDirectory(directory);
        Mesh mesh = SceneMeshReader.Read(File.ReadAllBytes(meshPath), SceneMeshReader.FormatOf(meshPath));
        mesh = MeshCleaner.Clean(mesh, new MeshCleanOptions
        {
            WeldVertices = true, RemoveDegenerateFaces = true, RemoveDuplicateFaces = true, RepairNonManifoldEdges = false, SmallComponentAreaRatio = 0, HoleFillMaxPerimeter = 0, UnifyOrientation = false, OrientOutward = false,
        });
        mesh = UvUnwrapper.Unwrap(mesh, new UvUnwrapOptions { TextureSize = texture }).Mesh;
        Write(directory, "verts.f32", mesh.Vertices);
        Write(directory, "uvs.f32", mesh.Uvs!);
        Write(directory, "idx.i32", mesh.Indices);

        byte[] rgb = File.ReadAllBytes(referencePath);
        int side = (int)Math.Sqrt(rgb.Length / 3);
        Assert.Equal(side * side * 3, rgb.Length);
        PaintViewCandidate[] views = [.. PaintViewCandidate.Fixed];
        for (int v = 0; v < views.Length; v++)
        {
            float[] image = new float[resolution * resolution * 3];
            for (int y = 0; y < resolution; y++)
            {
                for (int x = 0; x < resolution; x++)
                {
                    int sourceX = (x + v * 37) % side, sourceY = y * side / resolution;
                    for (int c = 0; c < 3; c++)
                        image[(y * resolution + x) * 3 + c] = Math.Min(1f, rgb[(sourceY * side + sourceX) * 3 + c] / 255f * (0.6f + 0.08f * v));
                }
            }

            Write(directory, $"view{v}.f32", image);
        }

        File.WriteAllText(Path.Combine(directory, "meta.json"), JsonSerializer.Serialize(new
        {
            tex = texture, res = resolution, views = views.Select(view => new { elev = view.Elevation, azim = view.Azimuth, weight = view.Weight }),
        }));
        output.WriteLine($"wrote {mesh.TriangleCount} faces, {views.Length} views to {directory}");
    }

    private static void Write<T>(string directory, string name, T[] values)
        where T : struct
    {
        byte[] bytes = new byte[System.Runtime.InteropServices.Marshal.SizeOf<T>() * values.Length];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(Path.Combine(directory, name), bytes);
    }

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
        output.WriteLine($"{surface.Count} texels, edge detector in the reference: {(realCanny ? "OpenCV Canny" : "a numpy stand-in (the sets of painted texels are not compared)")}");

        int viewCount = f.Views.Length / 3;
        bool[][] ours = new bool[viewCount][], theirs = new bool[viewCount][];
        for (int v = 0; v < viewCount; v++)
        {
            float[] image = f.Archive.Get($"image{v}").ToFloat32();
            PaintViewBake bake = PaintBackProjector.Project(renderer, surface, image, f.Resolution, 3, f.Views[v * 3], f.Views[v * 3 + 1]);
            float[] cosine = At(f.Archive.Get($"cos{v}").ToFloat32(), f.Texels, 1), color = At(f.Archive.Get($"color{v}").ToFloat32(), f.Texels, 3);
            ours[v] = [.. bake.Weights.Select(w => w > 0f)];
            theirs[v] = [.. cosine.Select(w => w > 0f)];
            CompareView(v, bake, cosine, color, realCanny);
            merger.Add(bake, f.Views[v * 3 + 2]);
        }

        (float[] texture, bool[] painted) = merger.Finish();
        float[] merged = f.Archive.Get("merged").ToFloat32();
        bool[] referencePainted = f.Archive.Get("painted").ToFloat32().Select(x => x > 0.5f).ToArray();
        List<float> actual = [], expected = [], actualAgreeing = [], expectedAgreeing = [];
        int onlyOurs = 0, onlyReference = 0;
        for (int i = 0; i < f.Texels.Length; i++)
        {
            int t = f.Texels[i];
            bool agree = Enumerable.Range(0, viewCount).All(v => ours[v][i] == theirs[v][i]);
            if (painted[t] && referencePainted[t])
            {
                for (int c = 0; c < 3; c++)
                {
                    actual.Add(texture[t * 3 + c]);
                    expected.Add(merged[t * 3 + c]);
                    if (agree)
                    {
                        actualAgreeing.Add(texture[t * 3 + c]);
                        expectedAgreeing.Add(merged[t * 3 + c]);
                    }
                }
            }
            else if (painted[t])
                onlyOurs++;
            else if (referencePainted[t])
                onlyReference++;
        }

        ParityResult all = ParityCompare.Compare("merged, all texels painted in both", actual.ToArray(), expected.ToArray());
        ParityResult agreeing = ParityCompare.Compare("merged, texels every view paints alike", actualAgreeing.ToArray(), expectedAgreeing.ToArray());
        output.WriteLine($"merged texture: painted in both {all.Count / 3}, only here {onlyOurs}, only upstream {onlyReference}");
        output.WriteLine($"{all.Name}: largest abs {all.MaxAbs:E2}, {all.Failures} of {all.Count} outside tolerance");
        output.WriteLine($"{agreeing.Name}: {agreeing.Count / 3} texels, largest abs {agreeing.MaxAbs:E2}, {agreeing.Failures} outside tolerance");
        // Where every view paints a texel the same way the merge is the same weighted mean; elsewhere a texel that only upstream paints in one view (the edge-mask margin) changes the mean.
        Assert.True(agreeing.Failures <= Math.Max(9, agreeing.Count / 1000), $"{agreeing.Name}: {agreeing.Failures} of {agreeing.Count} outside tolerance, largest {agreeing.MaxAbs}");
        Assert.True(onlyOurs <= 20, $"{onlyOurs} texels are painted only here");
        if (realCanny)
        {
            // Measured: 1. A wrong edge mask moves thousands of texels (the missing non-maximum suppression cost 2,093 in one view).
            Assert.True(onlyReference <= 10, $"{onlyReference} texels are painted only upstream");
            ParityResult everything = ParityCompare.Compare("merged, all texels painted in both", actual.ToArray(), expected.ToArray());
            Assert.True(everything.Failures <= Math.Max(30, everything.Count / 2000), $"{everything.Failures} of {everything.Count} merged values are outside tolerance");
        }
    }

    private void CompareView(int view, PaintViewBake bake, float[] cosine, float[] color, bool realCanny)
    {
        List<float> weightsHere = [], weightsThere = [], colorsHere = [], colorsThere = [];
        int onlyOurs = 0, onlyReference = 0, both = 0;
        for (int i = 0; i < cosine.Length; i++)
        {
            bool here = bake.Weights[i] > 0f, there = cosine[i] > 0f;
            if (here && there)
            {
                both++;
                weightsHere.Add(bake.Weights[i]);
                weightsThere.Add(cosine[i]);
                for (int c = 0; c < 3; c++)
                {
                    colorsHere.Add(bake.Colors[i * 3 + c]);
                    colorsThere.Add(color[i * 3 + c]);
                }
            }
            else if (here)
                onlyOurs++;
            else if (there)
                onlyReference++;
        }

        ParityResult weights = ParityCompare.Compare($"view {view} weight", weightsHere.ToArray(), weightsThere.ToArray());
        ParityResult colors = ParityCompare.Compare($"view {view} colour", colorsHere.ToArray(), colorsThere.ToArray());
        output.WriteLine($"view {view}: painted in both {both}, only here {onlyOurs}, only upstream {onlyReference}; weight largest abs {weights.MaxAbs:E2} ({weights.Failures} outside tolerance); colour largest abs {colors.MaxAbs:E2} ({colors.Failures} outside tolerance)");
        Assert.True(colors.Passed, $"{colors.Name}: {colors.Failures} of {colors.Count} outside tolerance, largest {colors.MaxAbs}");
        // A texel on the edge between two faces can resolve to the neighbouring face in one rasterizer and not the other (a float tie), which changes its weight: allow one in a thousand.
        Assert.True(weights.Failures <= Math.Max(3, weights.Count / 1000), $"{weights.Name}: {weights.Failures} of {weights.Count} outside tolerance, largest {weights.MaxAbs}");
        Assert.True(onlyOurs <= 5, $"view {view}: {onlyOurs} texels are painted only here");
        if (realCanny)
            Assert.True(onlyReference <= 10, $"view {view}: {onlyReference} texels are painted only upstream"); // measured: 0 to 2
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
    public void NormalAndPositionMapsMatchUpstreamRendersOfTheSameViews()
    {
        if (!TryLoad(out Loaded f))
            return;
        if (!f.Archive.Has("normal0"))
        {
            const string reason = "the fixture has no geometry maps (regenerate it with the current dump_paint_bake.py)";
            if (Environment.GetEnvironmentVariable(ParityFixtures.RequireWeightsVariable) == "1")
                throw new InvalidOperationException($"{ParityFixtures.RequireWeightsVariable}=1 but {reason}.");
            output.WriteLine($"SKIPPED: {reason}.");
            return;
        }

        PaintViewRenderer renderer = new(PaintMeshFrame.Create(f.Mesh));
        for (int v = 0; v < f.Views.Length / 3; v++)
        {
            foreach ((string name, byte[] bytes) in new[]
            {
                ("normal", renderer.RenderNormalMap(f.Views[v * 3], f.Views[v * 3 + 1], f.Resolution)),
                ("position", renderer.RenderPositionMap(f.Views[v * 3], f.Views[v * 3 + 1], f.Resolution)),
            })
            {
                float[] reference = f.Archive.Get($"{name}{v}").ToFloat32();
                Assert.Equal(reference.Length, bytes.Length);
                int different = 0, large = 0, worst = 0;
                for (int i = 0; i < bytes.Length; i++)
                {
                    int expected = Math.Clamp((int)(reference[i] * 255f), 0, 255), difference = Math.Abs(expected - bytes[i]);
                    if (difference > 0)
                        different++;
                    if (difference > 2)
                        large++;
                    worst = Math.Max(worst, difference);
                }

                output.WriteLine($"view {v} {name} map: {different} of {bytes.Length} values differ ({100.0 * different / bytes.Length:F4} %), {large} by more than 2 of 255 (largest {worst})");
                // A value moves by a level where float rounding crosses a quantisation step; a pixel on the edge between two faces can resolve to the neighbour in one rasterizer and not the other (a float tie),
                // which moves its three values a lot. A different convention or pixel assignment would move thousands.
                Assert.True(different <= bytes.Length / 1000 && large <= bytes.Length / 5000, $"view {v} {name}: {different} values differ, {large} by more than 2");
            }
        }
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
