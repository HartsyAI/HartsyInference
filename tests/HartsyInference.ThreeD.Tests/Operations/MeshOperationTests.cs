using System.Numerics;
using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Operations;

namespace HartsyInference.ThreeD.Tests.Operations;

/// <summary>The operations on small synthetic models: what each does, what it keeps (UVs, normals, textures, seams), what it reports losing, and the validation around them.</summary>
public sealed class MeshOperationTests
{
    private static MeshImage Png() => new("image/png", PngTextureEncoder.Encode(new TextureImage(2, 2, 3, [.. Enumerable.Range(0, 12).Select(i => (byte)(i * 20))])));

    /// <summary>A UV sphere. Like a real textured mesh it stores the seam column and the poles more than once, so the vertex count is larger than the number of surface points.</summary>
    private static Mesh Sphere(int rings, int segments, float radius = 1f, float noise = 0f, float centreX = 0f)
    {
        List<float> v = [], n = [], uv = [];
        List<int> idx = [];
        Random random = new(7);
        float[] radii = new float[(rings + 1) * segments];
        for (int i = 0; i < radii.Length; i++)
            radii[i] = radius * (1f + noise * ((float)random.NextDouble() - 0.5f));
        for (int r = 0; r <= rings; r++)
        {
            double phi = Math.PI * r / rings;
            for (int s = 0; s <= segments; s++)
            {
                double th = 2 * Math.PI * (s % segments) / segments;
                float rad = r == 0 || r == rings ? radius : radii[r * segments + s % segments];
                bool pole = r == 0 || r == rings;
                float x = pole ? 0f : (float)(Math.Sin(phi) * Math.Cos(th)), y = pole ? (r == 0 ? 1f : -1f) : (float)Math.Cos(phi), z = pole ? 0f : (float)(Math.Sin(phi) * Math.Sin(th));
                v.AddRange([centreX + x * rad, y * rad, z * rad]);
                n.AddRange([x, y, z]);
                uv.AddRange([s / (float)segments, r / (float)rings]);
            }
        }
        for (int r = 0; r < rings; r++)
        {
            for (int s = 0; s < segments; s++)
            {
                int a = r * (segments + 1) + s, b = a + segments + 1;
                idx.AddRange([a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        return new Mesh { Vertices = [.. v], Indices = [.. idx], Normals = [.. n], Uvs = [.. uv] };
    }

    /// <summary>The mesh without its last <paramref name="count"/> indices, which opens a hole where those triangles were.</summary>
    private static Mesh Trim(Mesh mesh, int count) => new() { Vertices = mesh.Vertices, Indices = mesh.Indices[..^count], Normals = mesh.Normals, Uvs = mesh.Uvs };

    private static MeshDocument Textured(params Mesh[] meshes) => new()
    {
        Primitives = [.. meshes.Select(m => new MeshPrimitive { Geometry = m, Material = 0 })],
        Materials = [new MeshMaterial { Name = "skin", BaseColor = Png(), BaseColorFactor = [0.8f, 0.4f, 0.2f, 1f] }],
        AssetExtrasJson = """{"hartsyinference":{"seed":3}}""",
    };

    private static MeshOperationRun Run(MeshDocument document, string operation, params (string Name, double Value)[] parameters) =>
        MeshOperations.Run(document, [new MeshOperationStep(operation, parameters.ToDictionary(p => p.Name, p => p.Value))]);

    private static float[] Bounds(MeshDocument document)
    {
        (Vector3 min, Vector3 max) = RecenterBounds(document);
        return [min.X, min.Y, min.Z, max.X, max.Y, max.Z];
    }

    private static (Vector3, Vector3) RecenterBounds(MeshDocument document)
    {
        Vector3 min = new(float.PositiveInfinity), max = new(float.NegativeInfinity);
        foreach (MeshPrimitive p in document.Primitives)
        {
            for (int i = 0; i < p.Geometry.Vertices.Length; i += 3)
            {
                Vector3 v = new(p.Geometry.Vertices[i], p.Geometry.Vertices[i + 1], p.Geometry.Vertices[i + 2]);
                min = Vector3.Min(min, v);
                max = Vector3.Max(max, v);
            }
        }
        return (min, max);
    }

    [Fact]
    public void DecimateSharesTheBudgetByPrimitiveSizeReportsTheLostTexturesAndLeavesAModelWithinTheTargetAlone()
    {
        Mesh big = Sphere(40, 80), small = Sphere(10, 12, centreX: 5f);
        MeshDocument document = Textured(big, small);

        MeshOperationRun run = Run(document, "decimate", ("triangles", 1500));
        MeshOperationRun within = Run(document, "decimate", ("triangles", 100000));

        MeshPrimitive reduced = run.Document.Primitives[0], part = run.Document.Primitives[1];
        Assert.InRange(run.Document.TriangleCount, 1000, 1600);
        Assert.InRange(reduced.Geometry.TriangleCount, 1000, 1500);
        Assert.True(part.Geometry.TriangleCount >= 1, "Every part keeps some triangles.");
        Assert.Null(reduced.Geometry.Uvs);
        Assert.NotNull(reduced.Geometry.Normals);
        Assert.Equal(reduced.Geometry.Vertices.Length, reduced.Geometry.Normals!.Length);
        Assert.False(run.Document.Materials[reduced.Material].Images.Any());
        Assert.Equal(0.8f, run.Document.Materials[reduced.Material].BaseColorFactor[0]);
        Assert.Same(big, within.Document.Primitives[0].Geometry);
        Assert.Same(small, within.Document.Primitives[1].Geometry);
        Assert.Empty(within.Steps[0].Notes);
        Assert.Contains(run.Steps[0].Notes, n => n.Contains("textures were dropped"));
        Assert.Equal(document.TriangleCount, run.Steps[0].TrianglesBefore);
        Assert.Equal(run.Document.TriangleCount, run.Steps[0].TrianglesAfter);
        Assert.Contains("seed", run.Document.AssetExtrasJson);
    }

    [Fact]
    public void DecimateByRatioAndTheExactlyOneRule()
    {
        MeshDocument document = Textured(Sphere(30, 60));

        MeshOperationRun half = Run(document, "decimate", ("ratio", 0.5));

        Assert.InRange(half.Document.TriangleCount, document.TriangleCount * 0.45, document.TriangleCount * 0.55);
        Assert.Throws<ArgumentException>(() => Run(document, "decimate"));
        Assert.Throws<ArgumentException>(() => Run(document, "decimate", ("triangles", 100), ("ratio", 0.5)));
    }

    [Fact]
    public void FillHolesClosesAnOpenBoxButNeverMistakesAUvSeamForAHole()
    {
        Mesh sphere = Sphere(12, 16);
        Mesh open = new()
        {
            Vertices = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
            Indices = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3],
        };

        MeshOperationRun seamed = Run(Textured(sphere), "fill-holes");
        MeshOperationRun filled = Run(new MeshDocument { Primitives = [new MeshPrimitive { Geometry = open }] }, "fill-holes", ("maxPerimeter", 3));
        MeshOperationRun tooBig = Run(new MeshDocument { Primitives = [new MeshPrimitive { Geometry = open }] }, "fill-holes", ("maxPerimeter", 0.5));

        Assert.Same(sphere, seamed.Document.Primitives[0].Geometry);
        Assert.NotNull(seamed.Document.Primitives[0].Geometry.Uvs);
        Assert.Empty(seamed.Steps[0].Notes);
        // The open top is a four-corner loop: a fan around a new centre vertex adds four triangles.
        Assert.Equal(open.TriangleCount + 4, filled.Document.TriangleCount);
        Assert.Contains(filled.Steps[0].Notes, n => n.Contains("4 triangle(s) added"));
        Assert.Equal(open.TriangleCount, tooBig.Document.TriangleCount);
    }

    [Fact]
    public void FillHolesDropsTheTexturesOfAPrimitiveItFilledAndSaysSo()
    {
        Mesh whole = Sphere(12, 16);
        // Remove the two triangles of one quad in the middle of a ring: a four-corner hole away from the poles.
        int first = (6 * 16 + 3) * 6;
        Mesh open = new() { Vertices = whole.Vertices, Indices = [.. whole.Indices[..first], .. whole.Indices[(first + 6)..]], Normals = whole.Normals, Uvs = whole.Uvs };
        MeshDocument document = Textured(open);

        MeshOperationRun run = Run(document, "fill-holes", ("maxPerimeter", 2));

        Assert.True(run.Document.TriangleCount > open.TriangleCount);
        Assert.Null(run.Document.Primitives[0].Geometry.Uvs);
        Assert.Contains(run.Steps[0].Notes, n => n.Contains("textures were dropped"));
    }

    [Fact]
    public void RemoveSmallComponentsDropsAFloaterAndKeepsEveryAttributeOfWhatRemains()
    {
        Mesh sphere = Sphere(12, 16);
        int baseVertices = sphere.VertexCount;
        float[] vertices = [.. sphere.Vertices, 10, 10, 10, 10.01f, 10, 10, 10, 10.01f, 10];
        Mesh withFloater = new()
        {
            Vertices = vertices,
            Indices = [.. sphere.Indices, baseVertices, baseVertices + 1, baseVertices + 2],
            Normals = [.. sphere.Normals!, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            Uvs = [.. sphere.Uvs!, 0, 0, 1, 0, 0, 1],
        };

        MeshOperationRun run = Run(Textured(withFloater), "remove-small-components");

        Mesh kept = run.Document.Primitives[0].Geometry;
        Assert.Equal(sphere.TriangleCount, kept.TriangleCount);
        Assert.Equal(baseVertices, kept.VertexCount);
        Assert.Equal(kept.VertexCount * 2, kept.Uvs!.Length);
        Assert.Equal(kept.VertexCount * 3, kept.Normals!.Length);
        Assert.True(run.Document.Materials[run.Document.Primitives[0].Material].Images.Any(), "Textures are kept.");
        Assert.Contains(run.Steps[0].Notes, n => n.StartsWith("Removed 1 of 2 piece(s)"));
    }

    [Fact]
    public void RecomputeNormalsGivesUnitNormalsSharedAcrossTheSeamAndKeepsUvs()
    {
        Mesh sphere = Sphere(12, 16);
        sphere.Normals = new float[sphere.Vertices.Length];

        Mesh result = Run(Textured(sphere), "recompute-normals").Document.Primitives[0].Geometry;

        Assert.Same(sphere.Uvs, result.Uvs);
        for (int v = 0; v < result.VertexCount; v++)
            Assert.Equal(1f, new Vector3(result.Normals![v * 3], result.Normals[v * 3 + 1], result.Normals[v * 3 + 2]).Length(), 3);
        // The first and last column of a ring are the same surface point: their normals must agree.
        int ring = 5, last = 16;
        for (int c = 0; c < 3; c++)
            Assert.Equal(result.Normals![(ring * 17) * 3 + c], result.Normals[(ring * 17 + last) * 3 + c], 4);
    }

    [Fact]
    public void SmoothRemovesNoiseWithoutShrinkingTheModelOrTearingTheSeamAndKeepsTheTexture()
    {
        Mesh noisy = Sphere(24, 32, noise: 0.2f);
        MeshDocument document = Textured(noisy);

        MeshOperationRun run = Run(document, "smooth", ("iterations", 10));

        Mesh smoothed = run.Document.Primitives[0].Geometry;
        double Spread(Mesh m)
        {
            double[] radii = [.. Enumerable.Range(0, m.VertexCount).Select(v => new Vector3(m.Vertices[v * 3], m.Vertices[v * 3 + 1], m.Vertices[v * 3 + 2]).Length())];
            double mean = radii.Average();
            return Math.Sqrt(radii.Select(r => (r - mean) * (r - mean)).Average());
        }
        Assert.True(Spread(smoothed) < Spread(noisy) * 0.6, "The noise went down.");
        Assert.InRange(Enumerable.Range(0, smoothed.VertexCount).Average(v => new Vector3(smoothed.Vertices[v * 3], smoothed.Vertices[v * 3 + 1], smoothed.Vertices[v * 3 + 2]).Length()),
            Enumerable.Range(0, noisy.VertexCount).Average(v => new Vector3(noisy.Vertices[v * 3], noisy.Vertices[v * 3 + 1], noisy.Vertices[v * 3 + 2]).Length()) * 0.95, 1.1);
        Assert.Same(noisy.Uvs, smoothed.Uvs);
        Assert.True(run.Document.Materials[run.Document.Primitives[0].Material].Images.Any());
        for (int c = 0; c < 3; c++)
            Assert.Equal(smoothed.Vertices[(10 * 33) * 3 + c], smoothed.Vertices[(10 * 33 + 32) * 3 + c], 5);
    }

    [Fact]
    public void SmoothLeavesAnOpenBoundaryWhereItIs()
    {
        Mesh open = Sphere(12, 16);
        open = Trim(open, 16 * 6);

        Mesh result = Run(Textured(open), "smooth", ("iterations", 5)).Document.Primitives[0].Geometry;

        // The last ring is the open boundary: its positions must not have moved.
        for (int s = 0; s < 16; s++)
        {
            int v = 11 * 17 + s;
            for (int c = 0; c < 3; c++)
                Assert.Equal(open.Vertices[v * 3 + c], result.Vertices[v * 3 + c], 5);
        }
    }

    [Fact]
    public void RecenterMovesEveryPartTogetherAndCanStandTheModelOnTheFloor()
    {
        MeshDocument document = Textured(Sphere(8, 12, centreX: 3f), Sphere(8, 12, centreX: 6f));

        float[] centred = Bounds(Run(document, "recenter").Document);
        float[] grounded = Bounds(Run(document, "recenter", ("ground", 1)).Document);
        float[] zGround = Bounds(Run(document, "recenter", ("ground", 1), ("up", 1)).Document);

        Assert.Equal(0f, (centred[0] + centred[3]) / 2, 4);
        Assert.Equal(0f, (centred[1] + centred[4]) / 2, 4);
        Assert.Equal(0f, grounded[1], 4);
        Assert.Equal(0f, (grounded[0] + grounded[3]) / 2, 4);
        Assert.Equal(0f, zGround[2], 4);
        Assert.Equal(5f, centred[3] - centred[0], 3);
    }

    [Fact]
    public void RescaleByFactorOrToALongestSideAndTheExactlyOneRule()
    {
        MeshDocument document = Textured(Sphere(8, 12, radius: 2f));

        float[] byLength = Bounds(Run(document, "rescale", ("longestSide", 1)).Document);
        float[] byFactor = Bounds(Run(document, "rescale", ("factor", 0.5)).Document);

        Assert.Equal(1f, MathF.Max(byLength[3] - byLength[0], MathF.Max(byLength[4] - byLength[1], byLength[5] - byLength[2])), 3);
        Assert.Equal(2f, byFactor[3] - byFactor[0], 3);
        Assert.Throws<ArgumentException>(() => Run(document, "rescale"));
        Assert.Throws<ArgumentException>(() => Run(document, "rescale", ("factor", 2), ("longestSide", 1)));
    }

    [Fact]
    public void ParametersAreValidatedAgainstTheirDescriptionBeforeAnyStepRuns()
    {
        MeshDocument document = Textured(Sphere(8, 12));

        Assert.Contains("has no parameter 'sides'", Assert.Throws<ArgumentException>(() => Run(document, "smooth", ("sides", 1))).Message);
        Assert.Throws<ArgumentOutOfRangeException>(() => Run(document, "smooth", ("iterations", 500)));
        Assert.Contains("whole number", Assert.Throws<ArgumentException>(() => Run(document, "smooth", ("iterations", 2.5))).Message);
        Assert.Throws<ArgumentException>(() => Run(document, "smooth", ("lambda", double.NaN)));
        Assert.Contains("takes none", Assert.Throws<ArgumentException>(() => Run(document, "recompute-normals", ("x", 1))).Message);
        Assert.Contains("decimate", Assert.Throws<NotSupportedException>(() => Run(document, "explode")).Message);
        // A bad later step fails before the first one runs.
        Assert.Throws<ArgumentOutOfRangeException>(() => MeshOperations.Run(document, [new("recenter"), new("smooth", new Dictionary<string, double> { ["iterations"] = 0 })]));
        Assert.Throws<ArgumentException>(() => MeshOperations.Run(document, []));
        Assert.Throws<ArgumentException>(() => MeshOperations.Run(document, [.. Enumerable.Repeat(new MeshOperationStep("recenter"), MeshOperations.MaxSteps + 1)]));
    }

    [Fact]
    public void StepsRunInOrderEachOnTheResultOfTheLastAndTheInputIsNeverChanged()
    {
        MeshDocument document = Textured(Sphere(8, 12, radius: 2f, centreX: 5f));
        float[] before = [.. document.Primitives[0].Geometry.Vertices];

        MeshOperationRun run = MeshOperations.Run(document, [new("recenter"), new("rescale", new Dictionary<string, double> { ["longestSide"] = 1 })]);

        Assert.Equal(["recenter", "rescale"], run.Steps.Select(s => s.Operation));
        float[] bounds = Bounds(run.Document);
        Assert.Equal(1f, bounds[3] - bounds[0], 3);
        Assert.Equal(0f, (bounds[0] + bounds[3]) / 2, 4);
        Assert.Equal(before, document.Primitives[0].Geometry.Vertices);
    }

    [Fact]
    public void EveryOperationDescribesItsParametersAndANewOneNeedsOnlyTheRegistry()
    {
        Assert.Equal(
            ["remove-small-components", "fill-holes", "decimate", "smooth", "recompute-normals", "recenter", "rescale"],
            MeshOperations.All.Select(o => o.Id));
        foreach (IMeshOperation operation in MeshOperations.All)
        {
            Assert.False(string.IsNullOrWhiteSpace(operation.Description));
            Assert.All(operation.Parameters, p => Assert.True(p.Min <= p.Max && (p.Default is null || (p.Default >= p.Min && p.Default <= p.Max))));
        }
    }
}
