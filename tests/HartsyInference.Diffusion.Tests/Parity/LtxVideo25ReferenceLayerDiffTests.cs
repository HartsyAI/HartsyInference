using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Layer-diff of <see cref="LtxVideo25DiffusionDecoder"/> against ComfyUI's <c>NADiffusionDecoder</c>, the
/// reference implementation of this exact VAE. The decoder's output is right in composition but heavily noised, and
/// no structural inspection (block wiring, chunk order, patchify, pixel-shuffle, timestep embedding) found the cause,
/// so this pins where the two diverge numerically.
///
/// <para>Both sides consume the SAME file-supplied latent and noise, so the pipeline is out of the picture and a
/// divergence localizes to a decoder stage. Run the reference half first:
/// <c>benchmarks/ltx25_layerdiff/ref_dump.py &lt;dir&gt;</c> under the ComfyUI venv, then point <c>LTX25_REFDUMP</c>
/// here. That folder's README documents the deeper bisects and the two traps that waste a session.</para></summary>
public sealed unsafe class LtxVideo25ReferenceLayerDiffTests
{
    private static string? DumpDir => Environment.GetEnvironmentVariable("LTX25_REFDUMP");

    /// <summary>Reads a dumped tensor's shape from the reference run's <c>shapes.json</c>, so the harness follows
    /// whatever geometry the Python side used instead of hardcoding one.</summary>
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

    /// <summary>Relative L2 against the reference dump, plus the raw moments so a shape/layout mismatch is visible
    /// even when relL2 saturates.</summary>
    private static (double RelL2, double RefStd, double OurStd) Compare(string dir, string name, Tensor ours)
    {
        string path = Path.Combine(dir, name + ".f32");
        // Not every tap has a reference dump (the inner bisect only dumps some stages) — skip rather than throw.
        if (!File.Exists(path)) return (0.0, double.NaN, double.NaN);
        byte[] bytes = File.ReadAllBytes(path);
        long n = bytes.Length / sizeof(float);
        if (n != ours.ElementCount)
            return (double.NaN, double.NaN, double.NaN);
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
        if (dir is null || !Directory.Exists(dir)) return;   // tier-lint: guarded
        string ckpt = TestPaths.LtxVideo2.VideoVae25;
        if (!File.Exists(ckpt)) return;                      // tier-lint: guarded

        (LtxVideo2CheckpointConverter.ConvertedWeights weights, SafeTensorsLoader loader) =
            LtxVideo2CheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _loader = loader;
        using LtxVideo25DiffusionDecoder decoder = new LtxVideo25DiffusionDecoder();
        // Cast to F32 exactly as LtxVideo2Recipe does, and as the reference dump does — otherwise the checkpoint's
        // BF16 weights put a ~1.6e-3 floor under every comparison and mask what we are looking for.
        Dictionary<string, Tensor> f32 = [];
        foreach ((string key, Tensor t) in weights.VaeDiffusionDecoder)
            f32[key] = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        decoder.LoadWeights(f32);

        using CudaBackend backend = new CudaBackend(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
        long[] latentShape = ShapeOf(dir, "latent");
        int lt = (int)latentShape[2], lh = (int)latentShape[3], lw = (int)latentShape[4];
        using Tensor latent = LoadRaw(Path.Combine(dir, "latent.f32"), new TensorShape(latentShape));
        using Tensor noise = LoadRaw(Path.Combine(dir, "noise.f32"), decoder.NoiseShape(lt, lh, lw));

        // Optional: mirror our own tensors to disk so the divergence can be dissected in numpy (which rows, which
        // channels) rather than through summary statistics.
        string? ourDir = Environment.GetEnvironmentVariable("LTX25_OURDUMP");
        if (ourDir is not null) Directory.CreateDirectory(ourDir);
        void Mirror(string name, Tensor t)
        {
            if (ourDir is null) return;
            byte[] raw = new byte[t.ElementCount * sizeof(float)];
            fixed (byte* dst = raw) Buffer.MemoryCopy((void*)t.DataPointer, dst, raw.Length, raw.Length);
            File.WriteAllBytes(Path.Combine(ourDir, name + ".f32"), raw);
        }

        List<(string Name, double RelL2, double RefStd, double OurStd)> report = [];
        decoder.Tap = (name, t) =>
        {
            Mirror(name, t);
            (double rel, double refStd, double ourStd) = Compare(dir, name, t);
            report.Add((name, rel, refStd, ourStd));
        };

        // Optional inner bisect: compare the FIRST attention module's internals (stage 0, block 0) against a
        // separate reference dump, so a divergence can be pinned to qkv / norm / rope / na3d.
        string? attnDir = Environment.GetEnvironmentVariable("LTX25_REFATTN");
        HashSet<string> seen = [];
        if (attnDir is not null && Directory.Exists(attnDir))
        {
            LtxVideo25NeighborhoodAttention3d.Tap = (name, t) =>
            {
                if (!seen.Add(name)) return;   // first block only
                Mirror(name, t);
                (double rel, double refStd, double ourStd) = Compare(attnDir, name, t);
                report.Add((name, rel, refStd, ourStd));
            };
        }

        using Tensor pixels = decoder.Decode(backend, latent, noise);
        LtxVideo25NeighborhoodAttention3d.Tap = null;

        string table = string.Join("\n", report.Select(r =>
            $"  {r.Name,-20} relL2 {(double.IsNaN(r.RelL2) ? "SHAPE-MISMATCH" : r.RelL2.ToString("E3")),-14} " +
            $"ref std {r.RefStd:F5}  ours {r.OurStd:F5}"));
        (string Name, double RelL2, double RefStd, double OurStd) firstBad =
            report.FirstOrDefault(r => double.IsNaN(r.RelL2) || r.RelL2 > 5e-3);

        Assert.True(firstBad.Name is null,
            $"first divergence from the ComfyUI reference at '{firstBad.Name}'\n{table}");
    }
}
