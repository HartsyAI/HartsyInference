using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Video.Tests;

/// <summary>Parts A5/A6 gate: SeedVr2Dit vs ByteDance's NaDiT at a tiny seeded-random config. Run
/// (1) <c>&lt;seedvr2-venv&gt; Parity/seedvr2_transformer_parity_dump.py &lt;SeedVR-checkout&gt; $SEEDVR2_PARITY_DIR</c>,
/// then (2) this test. Per-block relL2 with first-divergence reporting; assert full output &lt; 1e-3
/// (fp32 ladder), log per stage at 1e-3. Skips cleanly when the dump dir is absent.</summary>
public sealed class SeedVr2DitParityTests
{
    // Must match Parity/seedvr2_transformer_parity_dump.py exactly.
    private const int VidDim = 128, TxtInDim = 32, Heads = 1, HeadDim = 128;
    private const int Layers = 4, MmLayers = 2;
    private const int T = 5, H = 90, W = 160, TxtLen = 7;
    private const float Timestep = 937.0f;

    private readonly ITestOutputHelper _output;

    public SeedVr2DitParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static SeedVr2Config TinyConfig => new()
    {
        VidDim = VidDim, TxtInDim = TxtInDim, EmbDim = 6 * VidDim, Heads = Heads, HeadDim = HeadDim,
        MlpDim = 512, NumLayers = Layers, MmLayers = MmLayers, InChannels = 33, OutChannels = 16,
    };

    [Fact]
    [Trait("Category", "Integration")]
    public void TinyConfig_ForwardMatchesReference_PerBlock()
    {
        string dir = Environment.GetEnvironmentVariable("SEEDVR2_PARITY_DIR") ?? "/tmp/seedvr2_parity";
        if (!File.Exists(Path.Combine(dir, "weights.safetensors")))
        {
            _output.WriteLine($"SKIPPED: no dump at {dir} — run seedvr2_transformer_parity_dump.py first.");
            return;
        }

        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(dir, "weights.safetensors"));
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        weights = new Dictionary<string, Tensor>(
            weights.Where(kv => !kv.Key.EndsWith(".rope.rope.freqs", StringComparison.Ordinal)));

        SeedVr2Dit dit = new(TinyConfig);
        dit.LoadWeights(weights);
        IBackend backend = new CpuBackend();

        Tensor latent = LoadBin(Path.Combine(dir, "input_latent.bin"), [T, H, W, 33]);
        Tensor txt = LoadBin(Path.Combine(dir, "input_txt.bin"), [TxtLen, TxtInDim]);

        string? firstDivergence = null;
        double worst = 0;
        dit.OnBlockOutput = (idx, vid, txtTok) =>
        {
            if (idx < 0)
                return;
            double vidRel = RelL2(vid, Path.Combine(dir, $"block{idx}_vid.bin"));
            double txtRel = RelL2(txtTok, Path.Combine(dir, $"block{idx}_txt.bin"));
            _output.WriteLine($"block{idx}: vid relL2 {vidRel:e2}, txt relL2 {txtRel:e2}");
            worst = Math.Max(worst, Math.Max(vidRel, txtRel));
            if (firstDivergence is null && (vidRel > 1e-3 || txtRel > 1e-3))
                firstDivergence = $"block{idx}";
        };

        Tensor output = dit.Forward(backend, latent, txt, Timestep);
        backend.Sync();
        double outRel = RelL2(output, Path.Combine(dir, "output.bin"));
        _output.WriteLine($"output: relL2 {outRel:e2}");
        if (firstDivergence is not null)
            _output.WriteLine($"FIRST DIVERGENCE at '{firstDivergence}' — the bug is in/before that stage.");
        Assert.True(outRel < 1e-3, $"final output relL2 {outRel:e2} exceeds 1e-3" +
            (firstDivergence is null ? "" : $" (first divergence: {firstDivergence})"));
    }

    private static Tensor LoadBin(string path, long[] shape)
    {
        byte[] bytes = File.ReadAllBytes(path);
        Tensor tensor = new Tensor(new TensorShape(shape), DType.F32);
        bytes.AsSpan().CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(tensor.AsSpan<float>()));
        return tensor;
    }

    private double RelL2(Tensor actual, string refPath)
    {
        float[] expected = new float[new FileInfo(refPath).Length / 4];
        using (FileStream fs = File.OpenRead(refPath))
            fs.ReadExactly(System.Runtime.InteropServices.MemoryMarshal.AsBytes(expected.AsSpan()));
        ReadOnlySpan<float> a = actual.AsSpan<float>();
        Assert.Equal(expected.Length, a.Length);
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - expected[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
        }
        return Math.Sqrt(num / (den + 1e-12));
    }
}
