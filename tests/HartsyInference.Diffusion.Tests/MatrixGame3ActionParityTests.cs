using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Isolated numeric parity for Matrix-Game 3.0's <see cref="MatrixGame3ActionModule"/> (mouse temporal
/// self-attn + keyboard cross-attn, the [8,28,28] θ=256 action RoPE) vs the upstream Skywork
/// <c>ActionModule.forward</c> (<c>dump_mg3_action_reference.py</c>). Runs the module alone on block-0's real weights
/// — no 30-block accumulation — so it directly gates the novel action surface. Gated on <c>MG3_DIT</c> +
/// <c>MG3_ACT_REF</c>; runs on CPU F32 (the true-math path). Skips cleanly when unset.</summary>
public sealed unsafe class MatrixGame3ActionParityTests
{
    private readonly ITestOutputHelper _out;
    public MatrixGame3ActionParityTests(ITestOutputHelper o) => _out = o;

    private static bool IsCuda => string.Equals(Environment.GetEnvironmentVariable("PARITY_BACKEND"), "cuda", StringComparison.OrdinalIgnoreCase);

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
    public void ActionModule_Forward_MatchReference()
    {
        string? ditPath = Environment.GetEnvironmentVariable("MG3_DIT");
        string? refPath = Environment.GetEnvironmentVariable("MG3_ACT_REF");
        if (ditPath is null || refPath is null || !File.Exists(ditPath) || !File.Exists(refPath)) return; // gated

        using IBackend backend = MakeBackend();
        using SafeTensorsLoader dl = new(); dl.Load(ditPath);

        // Extract just block-0's action weights (casting bf16 -> F32 for the CPU backend / true-math parity).
        const string prefix = "blocks.0.action_model.";
        Dictionary<string, Tensor> w = new();
        foreach ((string k, Tensor t) in dl.GetAllTensors())
            if (k.StartsWith(prefix, StringComparison.Ordinal))
                w[k] = IsCuda || t.DType == DType.F32 ? t : t.CastTo(DType.F32);

        using SafeTensorsLoader rl = new(); rl.Load(refPath);
        Tensor xRef = rl.GetTensor("x");          // [48, 3072]
        Tensor mouse = rl.GetTensor("mouse");     // [9, 2]
        Tensor keyboard = rl.GetTensor("keyboard"); // [9, 6]
        int s = (int)xRef.Shape[0], dim = (int)xRef.Shape[1];

        double worst = 1.0;
        foreach ((string name, bool m, bool k, string refName) in new[]
                 { ("mouse-only", true, false, "out_m"), ("both", true, true, "out") })
        {
            MatrixGame3ActionModule action = new(imgHiddenSize: 3072, enableMouse: m, enableKeyboard: k);
            action.LoadWeights(w, "blocks.0.action_model");
            Tensor hidden = new(new TensorShape(s, dim), DType.F32);
            new ReadOnlySpan<float>((float*)xRef.DataPointer, s * dim).CopyTo(new Span<float>((float*)hidden.DataPointer, s * dim));
            action.Forward(backend, hidden, (3, 4, 4), stackalloc[] { 0, 1, 2 }, memoryFrames: 0, mouse, keyboard);
            (double maxAbs, double corr, double relL2) = Compare(hidden, rl.GetTensor(refName));
            _out.WriteLine($"MG3 ActionModule {name,-13}: maxAbs={maxAbs:E3} corr={corr:F8} relL2={relL2:E3}");
            hidden.Dispose();
            worst = Math.Min(worst, corr);
        }
        Assert.True(worst > 0.9999, $"worst stream corr {worst}");
    }

    private static (double MaxAbs, double Corr, double RelL2) Compare(Tensor a, Tensor b)
    {
        long n = Math.Min(a.ElementCount, b.ElementCount);
        float* pa = (float*)a.DataPointer; float* pb = (float*)b.DataPointer;
        double maxAbs = 0, dot = 0, na = 0, nb = 0, err = 0;
        for (long i = 0; i < n; i++)
        {
            double x = pa[i], y = pb[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y; err += (x - y) * (x - y);
        }
        return (maxAbs, dot / (Math.Sqrt(na * nb) + 1e-12), Math.Sqrt(err) / (Math.Sqrt(nb) + 1e-12));
    }
}
