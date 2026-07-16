using System.Diagnostics;
using HartsyInference.Cuda;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.Trellis;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests;

/// <summary>TRELLIS end-to-end image→3D Gaussian splat, driven by a pre-computed DINOv2 cond (the C# DINOv2-reg
/// conditioner is the last remaining piece; here the cond is dumped from the reference). Runs the full C# generative
/// path: SS flow → SS decoder → active voxels → SLAT flow → denorm → GS decoder → to_representation → PLY.
/// Gated on the ckpts + <c>/tmp/trellis_ref/cond_*.safetensors</c>.</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class TrellisGenerationTests
{
    private readonly ITestOutputHelper _out;
    public TrellisGenerationTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void ImageTo3D_GeneratesSplat()
    {
        string dir = "/tmp/TRELLIS-weights/ckpts";
        string condFile = Environment.GetEnvironmentVariable("TRELLIS_COND") ?? "/tmp/trellis_ref/cond_dragon.safetensors";
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(TrellisGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(dir) || !File.Exists(condFile) || !Directory.Exists(ptx)) { _out.WriteLine("SKIPPED."); return; }
        int seed = int.Parse(Environment.GetEnvironmentVariable("TRELLIS_SEED") ?? "42");

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptx);
        backend.CacheWeightCasts = false;
        Stopwatch sw = Stopwatch.StartNew();

        SparseStructureFlow ssFlow = new(); ssFlow.LoadWeights(Load(dir, "ss_flow_img_dit_L_16l8_fp16"));
        SparseStructureDecoder ssDec = new(); ssDec.LoadWeights(Load(dir, "ss_dec_conv3d_16l8_fp16"));
        SlatFlowModel slatFlow = new(); slatFlow.LoadWeights(Load(dir, "slat_flow_img_dit_L_64l8p2_fp16"));
        SlatGaussianDecoder gsDec = new(); gsDec.LoadWeights(Load(dir, "slat_dec_gs_swin8_B_64l8gs32_fp16"));

        using SafeTensorsLoader cl = new(); cl.Load(condFile);
        Tensor mean = cl.GetTensor("mean"); Tensor std = cl.GetTensor("std");

        // Image cond: computed in-engine by the C# DINOv2-vitl14-reg conditioner when the remapped weights are
        // present (self-contained path); else fall back to the pre-computed reference cond. The conditioner is fed
        // the preprocessed input (`prep`) — the raw-image premultiply-alpha/LANCZOS crop is the one remaining
        // Python step (as with TripoSR/Hunyuan3D background removal).
        string dinoFile = "/tmp/trellis_ref/dinov2_vitl14_reg.safetensors";
        Tensor cond;
        if (File.Exists(dinoFile))
        {
            using SafeTensorsLoader dl = new(); dl.Load(dinoFile);
            TrellisImageConditioner conditioner = new(); conditioner.LoadWeights(dl.GetAllTensors());
            backend.PreloadWeights(conditioner.EnumerateWeights());
            cond = conditioner.Encode(backend, cl.GetTensor("prep"));
            backend.FreeWeights(conditioner.EnumerateWeights());
            _out.WriteLine($"cond computed in-engine (DINOv2-reg conditioner) {cond.Shape}");
        }
        else { cond = cl.GetTensor("cond"); _out.WriteLine("cond loaded pre-computed (dino weights absent)"); }
        Tensor negCond = new(cond.Shape, DType.F32); backend.Fill(negCond, 0f);
        _out.WriteLine($"loaded in {sw.Elapsed.TotalSeconds:F1}s");

        // ── Stage 1: sparse structure → active-voxel coords ──
        sw.Restart();
        backend.PreloadWeights(ssFlow.EnumerateWeights());
        Tensor noise1 = GaussianNoise(new TensorShape(new long[] { 1, 8, 16, 16, 16 }), seed);
        Tensor zS = new TrellisSparseStructureSampler().Sample(backend, ssFlow, noise1, cond, negCond);
        backend.FreeWeights(ssFlow.EnumerateWeights());
        backend.PreloadWeights(ssDec.EnumerateWeights());
        Tensor occ = ssDec.Decode(backend, zS);
        backend.FreeWeights(ssDec.EnumerateWeights());
        int[] coords = ActiveCoords(occ);
        int nv = coords.Length / 4;
        _out.WriteLine($"[stage1] {sw.Elapsed.TotalSeconds:F1}s → {nv} active voxels");
        Assert.True(nv > 100, $"stage-1 produced too few voxels ({nv})");

        // ── Stage 2: structured latent over the active voxels ──
        sw.Restart();
        backend.PreloadWeights(slatFlow.EnumerateWeights());
        Tensor slatNoise = GaussianNoise(new TensorShape(1, nv, 8), seed + 1);
        SparseTensor slat = new TrellisSlatSampler().Sample(backend, slatFlow, new SparseTensor(slatNoise, coords, 64), cond, negCond);
        backend.FreeWeights(slatFlow.EnumerateWeights());
        TrellisSlatSampler.Denormalize(slat, mean, std);
        _out.WriteLine($"[stage2] {sw.Elapsed.TotalSeconds:F1}s");

        // ── GS decode → splat ──
        sw.Restart();
        backend.PreloadWeights(gsDec.EnumerateWeights());
        SparseTensor gs = gsDec.Forward(backend, slat);
        backend.FreeWeights(gsDec.EnumerateWeights());
        GaussianSplatCloud cloud = TrellisGaussianRepresentation.Build(gs);
        _out.WriteLine($"[gs-decode] {sw.Elapsed.TotalSeconds:F1}s → {cloud.Count} gaussians");

        string outDir = TestPaths.OutputDir; Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, $"trellis_{Path.GetFileNameWithoutExtension(condFile)}_{DateTime.Now:yyyyMMdd_HHmmss}.ply");
        PlyWriter.SaveSplats(outPath, cloud);
        _out.WriteLine($"Saved: {outPath} ({new FileInfo(outPath).Length / 1024 / 1024} MB)");
        Assert.True(cloud.Count == nv * 32 && File.Exists(outPath));
    }

    private static IReadOnlyDictionary<string, Tensor> Load(string dir, string name)
    {
        SafeTensorsLoader l = new(); l.Load(Path.Combine(dir, name + ".safetensors")); return l.GetAllTensors();
    }

    private static int[] ActiveCoords(Tensor occ)   // occ [1,1,D,H,W] → active (b,x,y,z) where occ>0
    {
        int d = (int)occ.Shape[2], h = (int)occ.Shape[3], w = (int)occ.Shape[4];
        float* p = (float*)occ.DataPointer;
        List<int> c = new();
        for (int z = 0; z < d; z++) for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
            if (p[(z * h + y) * w + x] > 0f) { c.Add(0); c.Add(z); c.Add(y); c.Add(x); }
        return c.ToArray();
    }

    private static Tensor GaussianNoise(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32); float* p = (float*)t.DataPointer; Random r = new(seed);
        for (long i = 0; i < shape.ElementCount; i++)
        {
            double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
            p[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return t;
    }
}
