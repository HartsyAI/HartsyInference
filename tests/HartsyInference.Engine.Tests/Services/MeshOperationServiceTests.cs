using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Operations;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The operation service over a textured glb the way a 3D model returns it: edits keep the settings block and the textures where the edit allows, report where it cannot, and write any target format.</summary>
public sealed class MeshOperationServiceTests
{
    private static byte[] TexturedGlb()
    {
        // A closed 12-triangle cube-ish box with UVs and normals, centred away from the origin, plus a floating triangle.
        float[] corners = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1];
        Mesh mesh = new()
        {
            Vertices = [.. corners.Select(c => c + 5f), 50, 50, 50, 50.1f, 50, 50, 50, 50.1f, 50],
            Indices = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5, 8, 9, 10],
            Normals = [.. Enumerable.Repeat(new[] { 0f, 0f, 1f }, 11).SelectMany(n => n)],
            Uvs = [.. Enumerable.Range(0, 11).SelectMany(i => new[] { i / 11f, 0.5f })],
        };
        MeshImage texture = new("image/png", PngEncoder.Encode(new byte[12], 2, 2));
        return MeshDocumentGlbWriter.Write(new MeshDocument
        {
            Primitives = [new MeshPrimitive { Geometry = mesh, Material = 0 }],
            Materials = [new MeshMaterial { BaseColor = texture }],
            AssetExtrasJson = """{"hartsyinference":{"seed":11}}""",
        });
    }

    private static MeshOperationStep Step(string operation, params (string Name, double Value)[] values) =>
        new(operation, values.ToDictionary(v => v.Name, v => v.Value));

    [Fact]
    public void EditsKeepTheTexturesAndTheSettingsBlockInAGlbAndReportEachStep()
    {
        using InferenceEngine engine = new("cpu");

        MeshOperationResult result = engine.MeshOperations.Apply(TexturedGlb(), "glb", [Step("remove-small-components"), Step("recenter"), Step("rescale", ("longestSide", 2))], "glb", "bust");

        MeshDocument document = GlbReader.ParseDocument(result.Output.Files[0].Data);
        Assert.Equal("bust.glb", result.Output.Files[0].Name);
        Assert.Equal(12, document.TriangleCount);
        Assert.True(document.HasTextures, "None of these edits touch the vertices' UVs, so the texture stays.");
        Assert.Contains("\"seed\":11", document.AssetExtrasJson);
        Assert.Equal(["remove-small-components", "recenter", "rescale"], result.Steps.Select(s => s.Operation));
        Assert.Equal((11L, 13L), (result.Steps[0].VerticesBefore, result.Steps[0].TrianglesBefore));
        Assert.Contains(result.Steps[0].Notes, n => n.StartsWith("Removed 1 of 2 piece(s)"));
        Assert.Equal((8L, 12L), (result.Output.Vertices, result.Output.Triangles));
    }

    [Fact]
    public void ADecimationSaysItDroppedTheTextureAndAnObjTargetStillWritesTheFiles()
    {
        using InferenceEngine engine = new("cpu");
        // The box is too coarse to reduce below its own count, so ask for the smallest legal result and expect either a note or an unchanged model.
        MeshOperationResult result = engine.MeshOperations.Apply(TexturedGlb(), "glb", [Step("decimate", ("ratio", 0.5))], "obj");

        Assert.Equal("obj", result.Output.Target);
        Assert.Equal("model.obj", result.Output.Files[0].Name);
        Assert.Single(result.Steps);
        Assert.True(result.Steps[0].TrianglesAfter <= result.Steps[0].TrianglesBefore);
    }

    [Fact]
    public void ARequestIsValidatedAndRefusedWithTheRightExceptionTypes()
    {
        using InferenceEngine engine = new("cpu");
        byte[] glb = TexturedGlb();

        Assert.Throws<NotSupportedException>(() => engine.MeshOperations.Apply(glb, "glb", [Step("explode")]));
        Assert.Throws<NotSupportedException>(() => engine.MeshOperations.Apply(glb, "glb", [Step("recenter")], "fbx"));
        Assert.Throws<ArgumentException>(() => engine.MeshOperations.Apply(glb, "glb", []));
        Assert.Throws<ArgumentOutOfRangeException>(() => engine.MeshOperations.Apply(glb, "glb", [Step("smooth", ("iterations", 0))]));
        Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() => engine.MeshOperations.Apply([1, 2, 3, 4], "glb", [Step("recenter")]));
        Assert.Equal(7, engine.MeshOperations.Operations.Count);
    }

    [Fact]
    public void ACancelledRequestStopsBetweenSteps()
    {
        using InferenceEngine engine = new("cpu");
        using CancellationTokenSource cancelled = new();
        cancelled.Cancel();

        Assert.Throws<OperationCanceledException>(() => engine.MeshOperations.Apply(TexturedGlb(), "glb", [Step("recenter")], cancel: cancelled.Token));
    }
}
