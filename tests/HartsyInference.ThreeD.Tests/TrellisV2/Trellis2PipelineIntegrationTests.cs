using HartsyInference.Core.Backends.Cpu;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary>
/// The 512 pipeline type on the real checkpoints with a synthetic alpha disc. Integration lane and opt-in: set
/// <c>HARTSY_TRELLIS2_SMOKE=1</c>. On the CPU backend the 1.3B flows take a long time (hours at the shipped 12 steps), so this runs two steps.
/// It checks that the chain produces a finite, indexable mesh; it does not check that the mesh looks like anything or matches the Python reference.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "Slow")]
public sealed class Trellis2PipelineIntegrationTests
{
    private readonly ITestOutputHelper _output;

    public Trellis2PipelineIntegrationTests(ITestOutputHelper output) => _output = output;

    [Trellis2RealWeightsFact("HARTSY_TRELLIS2_SMOKE")]
    public void Grid512_ProducesAFiniteIndexedMeshFromARealCheckpointSet()
    {
        const int size = 128;
        byte[] rgb = new byte[size * size * 3];
        byte[] alpha = new byte[size * size];
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int i = y * size + x;
                bool inside = (x - 64) * (x - 64) + (y - 64) * (y - 64) < 40 * 40;
                rgb[i * 3] = (byte)(inside ? 180 : 0);
                rgb[i * 3 + 1] = (byte)(inside ? 120 : 0);
                rgb[i * 3 + 2] = (byte)(inside ? 60 : 0);
                alpha[i] = inside ? (byte)255 : (byte)0;
            }
        }

        using CpuBackend backend = new();
        using Trellis2ImageTo3DPipeline pipeline = Trellis2ImageTo3DPipeline.LoadFromPath(backend, Trellis2RealWeightsFactAttribute.FindFolder()!);
        Trellis2InputImage image = new() { Rgb = rgb, Alpha = alpha, Width = size, Height = size };

        Mesh mesh = pipeline.Generate(
            image,
            new Trellis2Options { Grid = 512, Steps = 2, Seed = 7 },
            p => _output.WriteLine($"{p.Step}/{p.TotalSteps} {p.ElapsedMs:F0} ms"));

        _output.WriteLine($"vertices={mesh.VertexCount} triangles={mesh.TriangleCount}");
        Assert.True(mesh.TriangleCount > 0);
        Assert.All(mesh.Vertices, v => Assert.True(float.IsFinite(v)));
        Assert.All(mesh.Indices, i => Assert.InRange(i, 0, mesh.VertexCount - 1));
    }
}
