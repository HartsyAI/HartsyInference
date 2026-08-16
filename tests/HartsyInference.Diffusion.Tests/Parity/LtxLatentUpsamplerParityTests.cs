using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Layer-diff of <see cref="LtxLatentUpsampler"/> against ComfyUI's
/// <c>comfy.ldm.lightricks.latent_upsampler.LatentUpsampler</c>, the reference implementation of this exact module.
/// It is the only genuinely new numerics in the LTX-2.5 two-stage pipeline, and a module that merely produces a
/// plausible latent would be invisible end to end.
///
/// <para>Both sides consume the SAME file-supplied latent. Run the reference half first —
/// <c>benchmarks/ltx2_latent_upsampler/ref_dump.py &lt;dir&gt;</c> under the ComfyUI venv — then point
/// <c>LTX25UPSAMPLER_REFDUMP</c> here.</para>
///
/// <para>Measured 2026-08-15 on the CPU backend, F32 both sides: relL2 rises monotonically from 8.8e-7 at
/// <c>initial_conv</c> to 3.4e-6 at <c>output</c> — the F32 accumulation floor over 18 convolutions, with every
/// stage's std matching to 5 decimals. 1e-5 is that floor with headroom; anything above it is a bug, not a
/// tolerance. Takes ~2 minutes: <c>mid_channels</c> is 1024 and the default Conv3d is a scalar gather loop.</para></summary>
public sealed unsafe class LtxLatentUpsamplerParityTests
{
    private static string? DumpDir => Environment.GetEnvironmentVariable("LTX25UPSAMPLER_REFDUMP");

    /// <summary>Stage taps in forward order; the reference dumps carry the same names.</summary>
    private static readonly string[] _stages =
    [
        "initial_conv", "initial_act", "res0", "res1", "res2", "res3",
        "up_conv", "up_shuffle", "post0", "post1", "post2", "post3", "output",
    ];

    private static long[] ShapeOf(string dir, string name)
    {
        using System.Text.Json.JsonDocument doc =
            System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "shapes.json")));
        return [.. doc.RootElement.GetProperty(name).EnumerateArray().Select(e => e.GetInt64())];
    }

    private static Tensor LoadRaw(string path, TensorShape shape)
    {
        byte[] bytes = File.ReadAllBytes(path);
        Tensor t = new Tensor(shape, DType.F32);
        if (bytes.Length != (int)shape.ElementCount * sizeof(float))
            throw new InvalidOperationException($"{path} is {bytes.Length} bytes, expected {shape.ElementCount * 4} for {shape}.");
        fixed (byte* src = bytes)
            Buffer.MemoryCopy(src, (void*)t.DataPointer, bytes.Length, bytes.Length);
        return t;
    }

    /// <summary>Relative L2 against the reference dump, plus the raw moments so a shape/layout mismatch stays
    /// visible when relL2 saturates.</summary>
    private static (double RelL2, double RefStd, double OurStd) Compare(string dir, string name, Tensor ours)
    {
        string path = Path.Combine(dir, name + ".f32");
        if (!File.Exists(path)) return (double.NaN, double.NaN, double.NaN);
        byte[] bytes = File.ReadAllBytes(path);
        long n = bytes.Length / sizeof(float);
        if (n != ours.ElementCount) return (double.NaN, double.NaN, double.NaN);
        float* o = (float*)ours.DataPointer;
        double num = 0, den = 0, refSum = 0, refSq = 0, ourSum = 0, ourSq = 0;
        fixed (byte* raw = bytes)
        {
            float* r = (float*)raw;
            for (long i = 0; i < n; i++)
            {
                double d = (double)r[i] - o[i];
                num += d * d; den += (double)r[i] * r[i];
                refSum += r[i]; refSq += (double)r[i] * r[i];
                ourSum += o[i]; ourSq += (double)o[i] * o[i];
            }
        }
        double refStd = Math.Sqrt(Math.Max(0, refSq / n - refSum / n * (refSum / n)));
        double ourStd = Math.Sqrt(Math.Max(0, ourSq / n - ourSum / n * (ourSum / n)));
        return (Math.Sqrt(num / Math.Max(den, 1e-30)), refStd, ourStd);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void MatchesTheComfyUiReferenceStageByStage()
    {
        string? dir = DumpDir;
        if (dir is null || !Directory.Exists(dir)) return;              // tier-lint: guarded
        string ckpt = TestPaths.LtxVideo2.LatentUpsampler25;
        if (!File.Exists(ckpt)) return;                                 // tier-lint: guarded

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(ckpt);
        // The checkpoint is BF16; leaving it there puts a ~1.6e-3 floor under every comparison, large enough to
        // hide a real defect. The pipeline casts to F32 too — Conv3d has no lower-precision path.
        Dictionary<string, Tensor> f32 = [];
        foreach ((string key, Tensor t) in loader.GetAllTensors())
            f32[key] = t.DType == DType.F32 ? t : t.CastTo(DType.F32);

        LtxLatentUpsampler upsampler = new LtxLatentUpsampler();
        upsampler.LoadWeights(f32, loader.Metadata?["config"]);

        using CpuBackend backend = new CpuBackend();
        using Tensor latent = LoadRaw(Path.Combine(dir, "latent.f32"), new TensorShape(ShapeOf(dir, "latent")));

        List<(string Name, double RelL2, double RefStd, double OurStd)> report = [];
        HashSet<string> tapped = [];
        upsampler.Tap = (name, t) =>
        {
            tapped.Add(name);
            (double rel, double refStd, double ourStd) = Compare(dir, name, t);
            report.Add((name, rel, refStd, ourStd));
        };
        using Tensor output = upsampler.Forward(backend, latent);
        upsampler.Tap = null;

        string table = string.Join("\n", report.Select(r =>
            $"  {r.Name,-14} relL2 {(double.IsNaN(r.RelL2) ? "SHAPE-MISMATCH" : r.RelL2.ToString("E3")),-14} " +
            $"ref std {r.RefStd:F5}  ours {r.OurStd:F5}"));
        string header = $"config: in={upsampler.InChannels} mid={upsampler.MidChannels} " +
            $"blocks={upsampler.NumBlocksPerStage} scale={upsampler.SpatialScale}\n";

        string[] missing = [.. _stages.Where(s => !tapped.Contains(s))];
        Assert.True(missing.Length == 0, $"stages never reached: {string.Join(", ", missing)}\n{header}{table}");

        // Tolerance is overridable so a passing run can still be made to print its table (set it below the floor).
        double tol = double.TryParse(Environment.GetEnvironmentVariable("LTX25UPSAMPLER_TOL"), out double t2) ? t2 : 1e-5;
        (string Name, double RelL2, double RefStd, double OurStd) firstBad =
            report.FirstOrDefault(r => double.IsNaN(r.RelL2) || r.RelL2 > tol);
        Assert.True(firstBad.Name is null,
            $"first divergence from the ComfyUI reference at '{firstBad.Name}'\n{header}{table}");
    }
}
