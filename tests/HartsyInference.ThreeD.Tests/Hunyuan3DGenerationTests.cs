using System.Diagnostics;
using HartsyInference.Cuda;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Vision.Codec;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests;

/// <summary>End-to-end Hunyuan3D-2 single-image → mesh on the GPU: DINOv2-giant conditioning → Flux flow-match
/// DiT (2-way CFG) → ShapeVAE occupancy → marching cubes → <c>.glb</c>. Gated on <c>HY3D_MODEL_DIR</c> (a dir
/// holding the single <c>hunyuan3d-dit-v2-0/model*.safetensors</c>, which bundles DiT+VAE+conditioner) +
/// <c>HY3D_IMAGE</c> (a foreground-on-gray PNG). <c>HY3D_STEPS</c>/<c>HY3D_GRID</c> tune speed (block glue is
/// CPU-resident until the GPU-residency perf pass, so keep steps/grid small for a first coherence check).</summary>
[Trait("Category", "Integration")]
public sealed class Hunyuan3DGenerationTests
{
    private readonly ITestOutputHelper _out;
    public Hunyuan3DGenerationTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Hunyuan3D_Gpu_ImageToMesh()
    {
        string? dir = Environment.GetEnvironmentVariable("HY3D_MODEL_DIR");
        string? image = Environment.GetEnvironmentVariable("HY3D_IMAGE");
        if (dir is null || !Directory.Exists(dir) || image is null || !File.Exists(image)) { _out.WriteLine("SKIPPED: HY3D_MODEL_DIR/HY3D_IMAGE unset."); return; }
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(Hunyuan3DGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptx)) { _out.WriteLine("SKIPPED: no PTX."); return; }
        int steps = int.Parse(Environment.GetEnvironmentVariable("HY3D_STEPS") ?? "5");
        int grid = int.Parse(Environment.GetEnvironmentVariable("HY3D_GRID") ?? "128");

        Stopwatch sw = Stopwatch.StartNew();
        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptx);
        backend.CacheWeightCasts = false;
        _out.WriteLine($"[1/3] Loading Hunyuan3D-2 from {dir}...");
        using Hunyuan3DShapePipeline pipeline = Hunyuan3DShapePipeline.LoadFromPath(backend, dir);
        _out.WriteLine($"  loaded in {sw.Elapsed.TotalSeconds:F1}s");

        (byte[] rgb, int w, int h) = PngDecoder.DecodeFromFile(image);
        _out.WriteLine($"[2/3] {Path.GetFileName(image)} = {w}x{h}, steps={steps}, grid={grid}");
        ImageTo3DRequest req = new() { ImageRgb = rgb, Width = w, Height = h, Steps = steps, GridResolution = grid, Seed = 0 };

        sw.Restart();
        ThreeDResult result = pipeline.Generate(req, p => _out.WriteLine($"  step {p.Step}/{p.TotalSteps} ({sw.Elapsed.TotalSeconds:F0}s)"));
        _out.WriteLine($"[3/3] Generated in {sw.Elapsed.TotalSeconds:F1}s");

        Mesh mesh = result.Mesh!;
        _out.WriteLine($"  mesh: {mesh.VertexCount} verts, {mesh.TriangleCount} tris");
        Assert.NotNull(result.Mesh);

        // Save the .glb first (always) so the artifact is available for inspection even if the assertion fails.
        if (mesh.TriangleCount > 0)
        {
            string outDir = TestPaths.OutputDir; Directory.CreateDirectory(outDir);
            string outPath = Path.Combine(outDir, $"hunyuan3d_{Path.GetFileNameWithoutExtension(image)}_{DateTime.Now:yyyyMMdd_HHmmss}.glb");
            GlbWriter.Save(outPath, mesh);
            _out.WriteLine($"  Saved: {outPath}");
        }
        Assert.True(mesh.TriangleCount > 100, $"degenerate mesh ({mesh.TriangleCount} tris)");
    }
}
