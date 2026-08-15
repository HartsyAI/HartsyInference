using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Layer-diff of <see cref="LtxVideo2VaeDecoder"/> (the LTX-2 CONVOLUTIONAL video VAE) against ComfyUI's
/// <c>causal_video_autoencoder.Decoder</c>, the reference implementation of this exact VAE. Its output is soft and
/// carries a periodic cross-hatch the diffusion decoder does not, and this decoder had never been layer-diffed.
///
/// <para>Both sides consume the SAME file-supplied latent, so the pipeline is out of the picture and a divergence
/// localizes to a decoder stage. Run the reference half first:
/// <c>benchmarks/ltx2_conv_layerdiff/ref_dump.py &lt;dir&gt;</c> under the ComfyUI venv, then point
/// <c>LTX2CONV_REFDUMP</c> here. That folder's README documents the two-run structure and the traps.</para>
///
/// <para>Runs on the CPU backend: the geometry is tiny and the decoder is backend-generic, so this needs no GPU.</para></summary>
public sealed unsafe class LtxVideo2ConvVaeReferenceLayerDiffTests
{
    private static string? DumpDir => Environment.GetEnvironmentVariable("LTX2CONV_REFDUMP");

    /// <summary>Stage taps in decode order. Reference dumps carry the same names.</summary>
    private static readonly string[] _stages =
    [
        "denorm", "conv_in", "mid_resnets",
        "up0_up", "up0_res", "up1_up", "up1_res", "up2_up", "up2_res", "up3_up", "up3_res",
        "norm_out", "conv_out", "pixels",
    ];

    private static long[] ShapeOf(string dir, string name)
    {
        using System.Text.Json.JsonDocument doc =
            System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "shapes.json")));
        return [.. doc.RootElement.GetProperty(name).EnumerateArray().Select(e => e.GetInt64())];
    }

    /// <summary>Renders the reference's own module list (dumped as <c>_structure</c>) in the same form as
    /// <c>LtxVideo2VaeDecoder.InferredStructure</c>: the reference's flat up_blocks 0/odd/even map to our
    /// mid block / upsamplers / per-stage resnets.</summary>
    private static string ReferenceStructure(string dir)
    {
        using System.Text.Json.JsonDocument doc =
            System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "shapes.json")));
        List<string> parts = [];
        foreach (System.Text.Json.JsonElement b in doc.RootElement.GetProperty("_structure").EnumerateArray())
        {
            int idx = b.GetProperty("idx").GetInt32();
            if (b.GetProperty("kind").GetString() == "res")
            {
                int n = b.GetProperty("num_layers").GetInt32();
                parts.Add(idx == 0 ? $"mid:{n}" : $"res{idx / 2 - 1}:{n}");
                continue;
            }
            long[] st = [.. b.GetProperty("stride").EnumerateArray().Select(e => e.GetInt64())];
            // Reference reduction factor is our channel upscale: conv_in / out both divide by it.
            parts.Add($"up{(idx - 1) / 2}:stride({st[0]},{st[1]},{st[2]}),upscale{b.GetProperty("reduction").GetInt32()}");
        }
        return string.Join(" ", parts);
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

    /// <summary>Relative L2 against the reference dump, plus the raw moments so a shape/layout mismatch is visible
    /// even when relL2 saturates.</summary>
    private static (double RelL2, double RefStd, double OurStd) Compare(string dir, string name, Tensor ours)
    {
        string path = Path.Combine(dir, name + ".f32");
        if (!File.Exists(path)) return (double.NaN, double.NaN, double.NaN);
        byte[] bytes = File.ReadAllBytes(path);
        long n = bytes.Length / sizeof(float);
        if (n != ours.ElementCount) return (double.NaN, double.NaN, double.NaN);
        Tensor f32 = ours.DType == DType.F32 ? ours : ours.CastTo(DType.F32);
        float* o = (float*)f32.DataPointer;
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
        if (!ReferenceEquals(f32, ours)) f32.Dispose();
        double refStd = Math.Sqrt(Math.Max(0, refSq / n - refSum / n * (refSum / n)));
        double ourStd = Math.Sqrt(Math.Max(0, ourSq / n - ourSum / n * (ourSum / n)));
        return (Math.Sqrt(num / Math.Max(den, 1e-30)), refStd, ourStd);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void MatchesTheComfyUiReferenceStageByStage()
    {
        string? dir = DumpDir;
        if (dir is null || !Directory.Exists(dir)) return;   // tier-lint: guarded
        string ckpt = TestPaths.LtxVideo2.VideoVae25Conv;
        if (!File.Exists(ckpt)) return;                      // tier-lint: guarded

        (LtxVideo2CheckpointConverter.ConvertedWeights weights, SafeTensorsLoader loader) =
            LtxVideo2CheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _loader = loader;

        // Cast the checkpoint's BF16 weights to F32 on BOTH sides — leaving ours in BF16 puts a ~1.6e-3 floor
        // under every comparison, large enough to hide a real defect.
        Dictionary<string, Tensor> f32 = [];
        foreach ((string key, Tensor t) in weights.Vae)
            f32[key] = t.DType == DType.F32 ? t : t.CastTo(DType.F32);

        float[]? mean = ToFloats(f32, "latents_mean");
        float[]? std = ToFloats(f32, "latents_std");
        LtxVideo2VaeDecoder decoder = new LtxVideo2VaeDecoder(
            patchSize: 4, isCausal: false, latentsMean: mean, latentsStd: std, computeDtype: DType.F32);
        decoder.LoadWeights(f32);

        using CpuBackend backend = new CpuBackend();
        long[] latentShape = ShapeOf(dir, "latent");
        using Tensor latent = LoadRaw(Path.Combine(dir, "latent.f32"), new TensorShape(latentShape));

        List<(string Name, double RelL2, double RefStd, double OurStd)> report = [];
        HashSet<string> tapped = [];
        decoder.Tap = (name, t) =>
        {
            tapped.Add(name);
            (double rel, double refStd, double ourStd) = Compare(dir, name, t);
            report.Add((name, rel, refStd, ourStd));
        };

        using Tensor pixels = decoder.Decode(backend, latent);
        decoder.Tap = null;

        string table = string.Join("\n", report.Select(r =>
            $"  {r.Name,-14} relL2 {(double.IsNaN(r.RelL2) ? "SHAPE-MISMATCH" : r.RelL2.ToString("E3")),-14} " +
            $"ref std {r.RefStd:F5}  ours {r.OurStd:F5}"));
        string header = $"structure: {decoder.InferredStructure}\nflags:     {decoder.InferredFlags}\n";

        // A stage that never fired is a silently-short structure walk, not a pass.
        string[] missing = [.. _stages.Where(s => !tapped.Contains(s))];
        Assert.True(missing.Length == 0, $"stages never reached: {string.Join(", ", missing)}\n{header}{table}");

        // CountResnets stops at the first missing index, so a converter mis-mapping would silently shrink a stage
        // and every remaining stage would still match in shape. Pin the inferred geometry to the reference's own.
        Assert.Equal(ReferenceStructure(dir), decoder.InferredStructure);

        // Tolerance is overridable so a passing run can still be made to print its table (set it below the floor).
        double tol = double.TryParse(Environment.GetEnvironmentVariable("LTX2CONV_TOL"), out double t2) ? t2 : 5e-3;
        (string Name, double RelL2, double RefStd, double OurStd) firstBad =
            report.FirstOrDefault(r => double.IsNaN(r.RelL2) || r.RelL2 > tol);
        Assert.True(firstBad.Name is null,
            $"first divergence from the ComfyUI reference at '{firstBad.Name}'\n{header}{table}");
    }

    private static float[]? ToFloats(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        if (!w.TryGetValue(key, out Tensor? t)) return null;
        Tensor f = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        float[] o = new float[f.ElementCount];
        new Span<float>((float*)f.DataPointer, o.Length).CopyTo(o);
        if (!ReferenceEquals(f, t)) f.Dispose();
        return o;
    }
}
