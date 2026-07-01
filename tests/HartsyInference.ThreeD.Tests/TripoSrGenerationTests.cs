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

/// <summary>End-to-end TripoSR single-image → mesh on the GPU. Loads the converted <c>triposr.safetensors</c>
/// (the F32 state dict dumped by <c>dump_triposr_reference.py</c>) via <see cref="TripoSrPipeline.LoadFromPath"/>,
/// runs the full DINO → triplane → NeRF → marching-cubes path on CUDA, and writes a <c>.glb</c> for visual
/// inspection. Gated on <c>TRIPOSR_WEIGHTS</c> (the safetensors) + <c>TRIPOSR_IMAGE</c> (a conditioning PNG,
/// e.g. <c>/tmp/TripoSR/examples/chair.png</c>). Skips cleanly when unset or when no GPU/PTX is present.</summary>
public sealed class TripoSrGenerationTests
{
    private readonly ITestOutputHelper _out;
    public TripoSrGenerationTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void TripoSr_Gpu_ImageToMesh()
    {
        string? weights = Environment.GetEnvironmentVariable("TRIPOSR_WEIGHTS");
        string? image = Environment.GetEnvironmentVariable("TRIPOSR_IMAGE");
        if (weights is null || !File.Exists(weights)) { _out.WriteLine("SKIPPED: TRIPOSR_WEIGHTS unset/missing."); return; }
        if (image is null || !File.Exists(image)) { _out.WriteLine("SKIPPED: TRIPOSR_IMAGE unset/missing."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(TripoSrGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _out.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
        _out.WriteLine($"[1/3] Loading TripoSR from {weights}...");
        using TripoSrPipeline pipeline = TripoSrPipeline.LoadFromPath(backend, weights);
        _out.WriteLine($"  loaded in {sw.Elapsed.TotalSeconds:F1}s");

        (byte[] rgb, int w, int h) = PngDecoder.DecodeFromFile(image);
        _out.WriteLine($"[2/3] Conditioning image {Path.GetFileName(image)} = {w}x{h}");

        ImageTo3DRequest req = new() { ImageRgb = rgb, Width = w, Height = h, GridResolution = 256, Seed = 0 };
        sw.Restart();
        ThreeDResult result = pipeline.Generate(req, p => _out.WriteLine($"  stage {p.Step}/{p.TotalSteps}"));
        _out.WriteLine($"[3/3] Generated in {sw.Elapsed.TotalSeconds:F1}s");

        Mesh mesh = result.Mesh!;
        _out.WriteLine($"  mesh: {mesh.VertexCount} verts, {mesh.TriangleCount} tris");
        Assert.NotNull(result.Mesh);
        Assert.True(mesh.TriangleCount > 100, $"degenerate mesh ({mesh.TriangleCount} tris) — check iso/checkpoint");
        Assert.Equal(mesh.VertexCount * 3, mesh.VertexColors!.Length);

        string outDir = TestPaths.OutputDir; Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, $"triposr_{Path.GetFileNameWithoutExtension(image)}_{DateTime.Now:yyyyMMdd_HHmmss}.glb");
        GlbWriter.Save(outPath, mesh);
        _out.WriteLine($"  Saved: {outPath}");
    }
}
