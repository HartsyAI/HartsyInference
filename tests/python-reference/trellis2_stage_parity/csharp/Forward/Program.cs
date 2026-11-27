using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Trellis;
using HartsyInference.ThreeD.Models.TrellisV2;

// usage: t2fwd <trellis2Folder> <captureDir> <ptxDir> <outFile> <backend cuda|cpu>
string folder = args[0], cap = args[1], ptx = args[2], outFile = args[3];
bool cuda = args.Length < 5 || args[4] == "cuda";
using IDisposable backendOwner = cuda ? new CudaBackend(0, ptx) : new HartsyInference.Core.Backends.Cpu.CpuBackend();
IBackend backend = (IBackend)backendOwner;

Trellis2ModelFiles files = Trellis2ModelFiles.Resolve(folder);
Trellis2PipelineConfig config = Trellis2PipelineConfig.Load(files.Pipeline);
Dictionary<string, (Array Data, long[] Dims)> flowCap = Tns.Read(Path.Combine(cap, "texture_flow.tns"));
List<(string, Array, long[])> outItems = [];

// 1. structure decoder on a seeded latent
using (Tensor z = SeedGenerator.CreateNoise(new TensorShape([1L, 8, 16, 16, 16]), 7))
{
    using SafeTensorsLoader loader = new();
    loader.Load(files.StructureDecoder);
    SparseStructureDecoder dec = new();
    dec.LoadWeights(loader.GetAllTensors());
    backend.PreloadWeights(dec.EnumerateWeights());
    Stopwatch sw = Stopwatch.StartNew();
    using Tensor occ = dec.Decode(backend, z);
    backend.Sync();
    double ms = sw.Elapsed.TotalMilliseconds;
    Console.WriteLine($"STAGE ss_dec: {ms / 1000:F2} s ({(cuda ? "cuda" : "cpu")})");
    outItems.Add(("ss_z", z.AsReadOnlySpan<float>().ToArray(), [1, 8, 16, 16, 16]));
    outItems.Add(("ss_out", occ.AsReadOnlySpan<float>().ToArray(), [1, 1, 64, 64, 64]));
    outItems.Add(("ss_ms", new[] { (float)ms }, [1]));
    backend.FreeWeights(dec.EnumerateWeights());
}

// 2. one texture-flow forward on the captured shape latent (normalized) and conditioning
float[] shapeFeats = (float[])flowCap["shape_feats"].Data;
int[] coords = (int[])flowCap["shape_coords"].Data;
int n = shapeFeats.Length / 32;
using Trellis2FlowWeights weights = Trellis2FlowWeights.Load(files.TextureFlow512!);
using Trellis2TextureFlow flow = new(weights);
using Tensor shapeNorm = new(new TensorShape(1, n, 32), DType.F32);
Span<float> sn = shapeNorm.AsSpan<float>();
for (int i = 0; i < n; i++)
    for (int c = 0; c < 32; c++)
        sn[i * 32 + c] = (shapeFeats[i * 32 + c] - config.ShapeLatentMean[c]) / config.ShapeLatentStd[c];
using Tensor x = SeedGenerator.CreateNoise(new TensorShape(1, n, flow.LatentChannels), 11);
float[] condData = (float[])flowCap["cond"].Data;
using Tensor cond = new(new TensorShape(1, condData.Length / 1024, 1024), DType.F32);
condData.AsSpan().CopyTo(cond.AsSpan<float>());
backend.PreloadWeights(flow.EnumerateWeights());
for (int rep = 0; rep < 2; rep++)
{
    Stopwatch sw = Stopwatch.StartNew();
    using Tensor v = flow.Forward(backend, x, shapeNorm, coords, 500f, cond);
    backend.Sync();
    double ms = sw.Elapsed.TotalMilliseconds;
    Console.WriteLine($"STAGE texture_flow_forward rep{rep}: {ms / 1000:F2} s, {n} tokens");
    if (rep == 1)
    {
        outItems.Add(("flow_out", v.AsReadOnlySpan<float>().ToArray(), [1, n, flow.LatentChannels]));
        outItems.Add(("flow_ms", new[] { (float)ms }, [1]));
    }
}
outItems.Add(("flow_x", x.AsReadOnlySpan<float>().ToArray(), [1, n, flow.LatentChannels]));
outItems.Add(("flow_shape_norm", shapeNorm.AsReadOnlySpan<float>().ToArray(), [1, n, 32]));
outItems.Add(("flow_coords", coords, [n, 4]));
outItems.Add(("flow_cond", condData, [condData.Length / 1024, 1024]));
Tns.Write(outFile, [.. outItems]);
Console.WriteLine("done");
