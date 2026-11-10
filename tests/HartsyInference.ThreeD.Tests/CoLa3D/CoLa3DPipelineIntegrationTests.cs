using System.Text;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Pipelines;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

/// <summary>
/// Full CoLa3D scene pipeline on a synthetic cube with the real checkpoints under Models/CoLa3D, Models/TRELLIS2-ShapeVAE and
/// Models/Utonia. Integration lane. When a checkpoint is missing the test is skipped, not passed; with
/// <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> a missing checkpoint fails the run.
/// </summary>
/// <remarks>Runs at a 64 grid: the CPU reference takes minutes at this size and would take hours at the 1024 default.</remarks>
[Trait("Category", "Integration")]
public sealed class CoLa3DPipelineIntegrationTests
{
    private const string RequireEnv = "HARTSY_REQUIRE_REAL_WEIGHTS";
    private const int TestResolution = 64;
    private const string CubeObj = "v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nv -1 1 -1\nv -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1\n"
        + "f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\nf 2 3 7\nf 2 7 6\nf 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n";

    private readonly ITestOutputHelper _output;

    public CoLa3DPipelineIntegrationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Cube_DecomposesIntoValidObjectMeshesWithRealWeights()
    {
        string? modelRoot = FindModelRoot();
        if (modelRoot is null)
        {
            const string message = "CoLa3D checkpoints (cola3d_free_q20.pt, TRELLIS2-ShapeVAE, utonia.pth) not found under a Models folder";
            if (Environment.GetEnvironmentVariable(RequireEnv) == "1")
                throw new InvalidOperationException($"{RequireEnv}=1 but {message}.");
            _output.WriteLine($"SKIPPED: {message}");
            return;
        }

        Mesh cube = SceneMeshReader.Read(Encoding.UTF8.GetBytes(CubeObj), "obj");
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = CoLa3DPipeline.LoadFromPath(backend, modelRoot);

        CoLa3DResult result = pipeline.Generate(cube, TestResolution);

        _output.WriteLine($"objects={result.Objects.Count} triangles={string.Join(",", result.Objects.Select(m => m.TriangleCount))}");
        Assert.NotEmpty(result.Objects);
        foreach (Mesh mesh in result.Objects)
        {
            Assert.True(mesh.TriangleCount > 0);
            Assert.True(mesh.VertexCount > 0);
            Assert.All(mesh.Vertices, v => Assert.True(float.IsFinite(v)));
            Assert.All(mesh.Indices, i => Assert.InRange(i, 0, mesh.VertexCount - 1));
        }
    }

    // Walks up from the test binary to the first folder holding every checkpoint this pipeline reads.
    private static string? FindModelRoot()
    {
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string models = Path.Combine(dir.FullName, "Models");
            if (File.Exists(Path.Combine(models, "CoLa3D", "cola3d_free_q20.pt"))
                && File.Exists(Path.Combine(models, "TRELLIS2-ShapeVAE", "shape_enc_next_dc_f16c32_fp16.safetensors"))
                && File.Exists(Path.Combine(models, "TRELLIS2-ShapeVAE", "shape_dec_next_dc_f16c32_fp16.safetensors"))
                && File.Exists(Path.Combine(models, "Utonia", "utonia.pth")))
            {
                return models;
            }

            dir = dir.Parent;
        }

        return null;
    }
}
