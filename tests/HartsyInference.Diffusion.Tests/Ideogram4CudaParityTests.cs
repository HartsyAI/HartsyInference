using System.Text.Json;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Runs the full <see cref="Ideogram4Transformer"/> forward on the CPU backend (reference) and the
/// CUDA backend (GPU-resident block path from Phase 3+), then compares the output velocity element-wise.
/// Uses the tiny synthetic checkpoint from <c>dump_ideogram4_full_forward.py</c> so it fits any GPU and needs
/// no real weights. Also reports the device-to-host sync count during the GPU forward — the residency metric
/// that should fall to ~0 once the transformer glue (Phase 4) and pipeline loop (Phase 5) are converted.
/// Skips cleanly when the synthetic reference is absent or CUDA is unavailable.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Ideogram4CudaParityTests
{
    private readonly ITestOutputHelper _output;
    public Ideogram4CudaParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Transformer_Forward_Cuda_Matches_Cpu()
    {
        string repoRoot = RepoRoot.Path;
        string refRoot = Path.Combine(repoRoot, "tests/python-reference/ideogram4_reference_tensors");
        string fwdDir = Path.Combine(refRoot, "full_forward");
        string inputsDir = Path.Combine(fwdDir, "inputs");
        string ckpt = Path.Combine(refRoot, "transformer.safetensors");

        if (!Directory.Exists(inputsDir) || !File.Exists(ckpt))
        {
            _output.WriteLine($"SKIPPED: synthetic reference not found at {refRoot}");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(inputsDir, "meta.json")));
        JsonElement m = meta.RootElement;
        int emb = m.GetProperty("emb").GetInt32();
        int layers = m.GetProperty("layers").GetInt32();
        int heads = m.GetProperty("heads").GetInt32();
        int ffn = m.GetProperty("ffn").GetInt32();
        int adaln = m.GetProperty("adaln").GetInt32();
        int inCh = m.GetProperty("in_ch").GetInt32();
        int llmDim = m.GetProperty("llm_dim").GetInt32();
        int theta = m.GetProperty("theta").GetInt32();
        int[] section = [.. m.GetProperty("section").EnumerateArray().Select(e => e.GetInt32())];
        float blockEps = m.GetProperty("block_eps").GetSingle();
        int L = m.GetProperty("L").GetInt32();
        float timestep = m.GetProperty("timestep").GetSingle();

        Ideogram4Config config = Ideogram4Config.V4 with
        {
            EmbDim = emb,
            NumLayers = layers,
            NumHeads = heads,
            IntermediateSize = ffn,
            AdalnDim = adaln,
            InChannels = inCh,
            LlmFeaturesDim = llmDim,
            RopeTheta = theta,
            MropeSection = (section[0], section[1], section[2]),
            NormEps = blockEps,
        };

        using SafeTensorsLoader loader = new();
        loader.Load(ckpt);
        Dictionary<string, Tensor> weights = new();
        foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
            weights[kvp.Key] = kvp.Value;

        using Ideogram4Transformer transformer = new(config);
        transformer.LoadWeights(weights);

        // CPU reference.
        float[] cpuVel;
        int n;
        using (CpuBackend cpu = new CpuBackend())
        {
            using Tensor x = LoadF32(Path.Combine(inputsDir, "x.bin"), 1, L, inCh);
            using Tensor llm = LoadF32(Path.Combine(inputsDir, "llm_features.bin"), 1, L, llmDim);
            using Tensor posIds = LoadF32(Path.Combine(inputsDir, "position_ids.bin"), 1, L, 3);
            int[] indicator = LoadI32(Path.Combine(inputsDir, "indicator.bin"), L);
            using Tensor vel = transformer.Forward(cpu, llm, x, timestep, posIds, indicator, null);
            n = (int)vel.ElementCount;
            cpuVel = new float[n];
            fixed (float* dst = cpuVel) Buffer.MemoryCopy((void*)vel.DataPointer, dst, n * 4L, n * 4L);
        }

        // CUDA forward (GPU-resident blocks). Materialize the result while the backend is alive.
        float[] cudaVel = new float[n];
        long syncs;
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(repoRoot, "src", "HartsyInference.Cuda", "Ptx");
        using (CudaBackend cuda = new CudaBackend(0, ptxDir))
        {
            using Tensor x = LoadF32(Path.Combine(inputsDir, "x.bin"), 1, L, inCh);
            using Tensor llm = LoadF32(Path.Combine(inputsDir, "llm_features.bin"), 1, L, llmDim);
            using Tensor posIds = LoadF32(Path.Combine(inputsDir, "position_ids.bin"), 1, L, 3);
            int[] indicator = LoadI32(Path.Combine(inputsDir, "indicator.bin"), L);
            cuda.ResetD2hSyncCount();
            using Tensor vel = transformer.Forward(cuda, llm, x, timestep, posIds, indicator, null);
            cuda.Sync();
            fixed (float* dst = cudaVel) Buffer.MemoryCopy((void*)vel.DataPointer, dst, n * 4L, n * 4L);
            syncs = cuda.GetD2hSyncCount();
        }

        double maxErr = 0, sumErr = 0;
        for (int i = 0; i < n; i++)
        {
            double e = Math.Abs(cpuVel[i] - cudaVel[i]);
            sumErr += e;
            if (e > maxErr) maxErr = e;
        }
        _output.WriteLine($"output_velocity[{n}]: avg_err={sumErr / n:E3}, max_err={maxErr:E3}");
        _output.WriteLine($"D2H syncs during GPU forward: {syncs} (Phase 4/5 should drive this toward ~0)");

        Assert.True(maxErr < 5e-3, $"CUDA transformer forward diverges from CPU: max_err {maxErr:E3}");
    }

    private static Tensor LoadF32(string path, params long[] dims)
    {
        TensorShape shape = dims.Length switch
        {
            2 => new TensorShape(dims[0], dims[1]),
            3 => new TensorShape(dims[0], dims[1], dims[2]),
            _ => throw new ArgumentException($"rank {dims.Length}"),
        };
        Tensor t = new(shape, DType.F32);
        byte[] raw = File.ReadAllBytes(path);
        long expected = shape.ElementCount * sizeof(float);
        if (raw.LongLength != expected)
            throw new InvalidDataException($"{path}: expected {expected} bytes, got {raw.LongLength}");
        fixed (byte* src = raw) Buffer.MemoryCopy(src, (void*)t.DataPointer, raw.LongLength, raw.LongLength);
        return t;
    }

    private static int[] LoadI32(string path, int count)
    {
        byte[] raw = File.ReadAllBytes(path);
        int[] result = new int[count];
        fixed (byte* src = raw) fixed (int* dst = result)
            Buffer.MemoryCopy(src, dst, raw.LongLength, raw.LongLength);
        return result;
    }
}
