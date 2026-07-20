using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Video.Tests;

/// <summary>Numerical parity harness (C# side) for the HunyuanVideo MM-DiT. Loads the tiny random weights + inputs
/// produced by <c>Parity/hunyuan_video_transformer_parity_dump.py</c> (diffusers reference,
/// <c>guidance_embeds=True</c>), runs <see cref="HunyuanVideoDit.Forward"/> with the SAME tiny config on the CPU
/// backend, and prints per-stage relL2 (post-<c>img_in</c>, post-refiner, each block, final velocity) vs the
/// reference .bin dumps. The first stage whose relL2 jumps to O(1) localizes the divergent op — this is how the
/// guidance-embedding / token-refiner / RoPE / final-mod-order bugs are caught fast.
///
/// <para>Run order: (1) <c>&lt;ComfyUI-venv python&gt; tests/HartsyInference.Video.Tests/Parity/hunyuan_video_transformer_parity_dump.py
/// $HYV_PARITY_DIR</c>, then (2) this test. Skips cleanly if the dump dir is absent.</para></summary>
public unsafe class HunyuanVideoDitParityTests
{
    private readonly ITestOutputHelper _output;
    public HunyuanVideoDitParityTests(ITestOutputHelper output) => _output = output;

    // Must match Parity/hunyuan_video_transformer_parity_dump.py exactly.
    private const int FLat = 2, HLat = 4, WLat = 4;         // post-patch: T=2, H=2, W=2 -> sImg = 8
    private const int SImg = 2 * 2 * 2;
    private const int L = 5;
    private const int InCh = 4, OutCh = 4, Inner = 16, Heads = 2, HeadDim = 8;
    private const int TextDim = 8, PooledDim = 6;
    private const int NumDouble = 1, NumSingle = 1;
    private const float Timestep = 500.0f, Guidance = 3000.0f;

    private static HunyuanVideoConfig TinyConfig() => new()
    {
        HiddenSize = Inner, NumHeads = Heads, NumDoubleBlocks = NumDouble, NumSingleBlocks = NumSingle,
        MlpDim = 4 * Inner, InChannels = InCh, OutChannels = OutCh, PatchSize = (1, 2, 2),
        RopeAxesDim = [4, 2, 2], RopeTheta = 256.0f, TextEmbedDim = TextDim, PooledEmbedDim = PooledDim,
        GuidanceEmbed = true, EmbeddedGuidanceScale = 3.0f,
    };

    [Fact]
    public void HunyuanVideo_Dit_PerStageParity_VsDiffusers()
    {
        string dir = Environment.GetEnvironmentVariable("HYV_PARITY_DIR") ?? "/tmp/hyv_parity";
        string weights = Path.Combine(dir, "weights.safetensors");
        if (!File.Exists(weights))
        {
            _output.WriteLine($"SKIPPED: parity dumps not found in {dir}. Run Parity/hunyuan_video_transformer_parity_dump.py first.");
            return;
        }

        Dictionary<string, float[]> refDump = new()
        {
            ["projin"] = ReadBin(Path.Combine(dir, "projin.bin")),
            ["refiner"] = ReadBin(Path.Combine(dir, "refiner.bin")),
            ["out_velocity"] = ReadBin(Path.Combine(dir, "out_velocity.bin")),
        };
        for (int i = 0; i < NumDouble; i++) refDump[$"double{i}"] = ReadBin(Path.Combine(dir, $"double{i}.bin"));
        for (int i = 0; i < NumSingle; i++) refDump[$"single{i}"] = ReadBin(Path.Combine(dir, $"single{i}.bin"));

        using SafeTensorsLoader loader = new();
        loader.Load(weights);
        using HunyuanVideoDit dit = new(TinyConfig());
        dit.LoadWeights(loader.GetAllTensors());

        Tensor latent = LoadTensor(Path.Combine(dir, "input_latent.bin"), [1, InCh, FLat, HLat, WLat]);
        Tensor txt = LoadTensor(Path.Combine(dir, "input_text.bin"), [1, L, TextDim]);
        Tensor pooled = LoadTensor(Path.Combine(dir, "input_pooled.bin"), [1, PooledDim]);

        CpuBackend backend = new();

        List<(string Stage, double RelL2)> results = new();
        dit.OnBlockOutput = (idx, img, txtTok) =>
        {
            if (idx == -1)
            {
                results.Add(("projin", RelL2(HostFloats(img), refDump["projin"])));
                results.Add(("refiner", RelL2(HostFloats(txtTok), refDump["refiner"])));
            }
            else
            {
                string stage = idx < NumDouble ? $"double{idx}" : $"single{idx - NumDouble}";
                results.Add((stage, RelL2(HostFloats(img), refDump[stage])));
            }
        };

        Tensor v = dit.Forward(backend, latent, txt, pooled, Timestep, Guidance);
        double outRel = RelL2(HostFloats(v), refDump["out_velocity"]);

        _output.WriteLine("HunyuanVideo DiT per-stage relL2 (C# vs diffusers, tiny matched config):");
        foreach ((string stage, double rel) in results)
            _output.WriteLine($"  {stage,-9} relL2 = {rel:E4}   {(rel < 1e-3 ? "OK" : "*** DIVERGENCE ***")}");
        _output.WriteLine($"  {"velocity",-9} relL2 = {outRel:E4}   {(outRel < 1e-3 ? "OK" : "*** DIVERGENCE ***")}");

        (string Stage, double RelL2) first = default;
        foreach ((string stage, double rel) in results) if (rel >= 1e-3) { first = (stage, rel); break; }
        if (first.Stage is not null)
            _output.WriteLine($"FIRST DIVERGENCE at stage '{first.Stage}' (relL2 {first.RelL2:E4}) — the bug is in/before that stage.");
        else if (outRel >= 1e-3)
            _output.WriteLine($"Blocks match but final velocity diverges (relL2 {outRel:E4}) — bug is in norm_out/proj_out/unpatchify/mod order.");
        else
            _output.WriteLine("FULL PARITY — the DiT is faithful to diffusers (guidance embedding, token refiner, RoPE, final mod).");

        v.Dispose(); latent.Dispose(); txt.Dispose(); pooled.Dispose();

        double worst = outRel;
        foreach ((_, double rel) in results) worst = Math.Max(worst, rel);
        Assert.True(worst < 1e-2, $"HunyuanVideo DiT diverges from diffusers (worst relL2 {worst:E4}); see per-stage log above.");
    }

    private static Tensor LoadTensor(string path, long[] shape)
    {
        float[] data = ReadBin(path);
        Tensor t = new(new TensorShape(shape), DType.F32);
        long n = t.Shape.ElementCount;
        if (data.Length != n) throw new InvalidDataException($"{Path.GetFileName(path)}: {data.Length} floats != {n}.");
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < n; i++) p[i] = data[i];
        return t;
    }

    private static float[] ReadBin(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, f.Length * 4);
        return f;
    }

    private static float[] HostFloats(Tensor t)
    {
        long n = t.Shape.ElementCount;
        float[] f = new float[n];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < n; i++) f[i] = p[i];
        return f;
    }

    private static double RelL2(float[] a, float[] b)
    {
        if (a.Length != b.Length) return double.NaN;
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++) { double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
        return Math.Sqrt(num / (den + 1e-12));
    }
}
