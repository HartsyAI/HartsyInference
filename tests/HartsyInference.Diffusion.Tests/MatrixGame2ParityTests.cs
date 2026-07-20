using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight numeric parity for Matrix-Game 2.0's Wan-backbone DiT forward (action module disabled)
/// vs the upstream Skywork <c>WanModel</c> reference. Gated on <c>MG2_DIT</c> (Skywork base safetensors) +
/// <c>MG2_REF</c> (<c>mg2_ref_io.safetensors</c> from <c>dump_mg2_reference.py</c>). <c>PARITY_BACKEND=cuda</c>
/// runs on the GPU. Skips cleanly when unset.</summary>
public sealed unsafe class MatrixGame2ParityTests
{
    private readonly ITestOutputHelper _out;
    public MatrixGame2ParityTests(ITestOutputHelper o) => _out = o;

    private static bool IsCuda => string.Equals(Environment.GetEnvironmentVariable("PARITY_BACKEND"), "cuda", StringComparison.OrdinalIgnoreCase);
    private static double AbsTol => IsCuda ? 0.3 : 5e-3;

    private static IBackend MakeBackend()
    {
        if (!IsCuda) return new CpuBackend();
        string? d = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && d is not null; i++, d = Path.GetDirectoryName(d))
        {
            string cand = Path.Combine(d, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(cand)) return new HartsyInference.Cuda.CudaBackend(0, cand);
        }
        return new HartsyInference.Cuda.CudaBackend(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
    }

    [Fact]
    public void WanBackbone_Forward_MatchReference()
    {
        string? ditPath = Environment.GetEnvironmentVariable("MG2_DIT");
        string? refPath = Environment.GetEnvironmentVariable("MG2_REF");
        if (ditPath is null || refPath is null || !File.Exists(ditPath) || !File.Exists(refPath)) return; // gated

        using IBackend backend = MakeBackend();
        using SafeTensorsLoader dl = new(); dl.Load(ditPath);
        using SafeTensorsLoader rl = new(); rl.Load(refPath);

        MatrixGame3CheckpointConverter.ConvertedWeights cw = MatrixGame2CheckpointConverter.Convert(dl.GetAllTensors());
        MatrixGame2Transformer model = new(MatrixGame2Config.Foundation);
        model.LoadWeights(cw.Transformer);

        // latent [1,36,3,16,16], clip [257,1280] (drop batch); single-timestep foundation → all frames equal.
        Tensor latent = rl.GetTensor("x36");
        Tensor clipB = rl.GetTensor("clip");
        int l = (int)clipB.Shape[1], cdim = (int)clipB.Shape[2];
        Tensor clip = new(new TensorShape(l, cdim), DType.F32);
        new ReadOnlySpan<float>((float*)clipB.DataPointer, l * cdim).CopyTo(new Span<float>((float*)clip.DataPointer, l * cdim));

        int gt = (int)latent.Shape[2];
        float[] timesteps = new float[gt];
        for (int i = 0; i < gt; i++) timesteps[i] = 500f;
        int[] ropeIdx = new int[gt];
        for (int i = 0; i < gt; i++) ropeIdx[i] = i;

        Dictionary<string, Tensor> taps = new();
        Tensor v = model.Forward(backend, latent, clip, timesteps, ropeIdx, gt, null, null, taps);
        foreach ((string tap, string refName) in new[] { ("patch", "tap_patch"), ("ctx", "tap_ctx"), ("block0", "tap_block0"), ("blockLast", "tap_blockLast") })
        {
            (double m2, double c2) = Compare(taps[tap], rl.GetTensor(refName));
            _out.WriteLine($"MG2 tap {tap,-10}: maxAbs={m2:E3} corr={c2:F8}");
        }
        Tensor refV = rl.GetTensor("v");
        (double maxAbs, double corr) = Compare(v, refV);
        _out.WriteLine($"MG2 Wan-backbone v: maxAbs={maxAbs:E3} corr={corr:F8}  (C# {v.Shape}, ref {refV.Shape})");
        v.Dispose(); clip.Dispose();
        Assert.True(corr > 0.9999, $"corr {corr}");
        Assert.True(maxAbs < AbsTol, $"maxAbs {maxAbs}");
    }

    private static (double, double) Compare(Tensor a, Tensor b)
    {
        long n = Math.Min(a.ElementCount, b.ElementCount);
        float* pa = (float*)a.DataPointer; float* pb = (float*)b.DataPointer;
        double maxAbs = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < n; i++)
        {
            double x = pa[i], y = pb[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y;
        }
        return (maxAbs, dot / (Math.Sqrt(na * nb) + 1e-12));
    }
}
