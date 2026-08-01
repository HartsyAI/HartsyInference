using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Video.Pipelines;
using HartsyInference.Vision.Codec;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Video.Tests;

/// <summary>Part A7 E2E gate: full C# restore of the Big Buck Bunny 360p clip vs the real-weight Python
/// reference (<c>run_seedvr2_e2e_reference.py</c>), with the reference's saved noises injected via
/// <see cref="SeedVr2RestorePipeline.NoiseHook"/> (torch RNG is unmatchable). FP32 both sides. Gate:
/// mean SSIM ≥ 0.995 on u8 frames; PSNR logged. Env: SEEDVR2_DIT / SEEDVR2_VAE / SEEDVR2_EMB /
/// SEEDVR2_E2E_REF / SEEDVR2_FRAMES (+ SEEDVR2_E2E_BACKEND=cuda|cpu, default cuda). Skips cleanly.</summary>
public sealed class SeedVr2E2eParityTests
{
    private readonly ITestOutputHelper _output;

    public SeedVr2E2eParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void BigBuckBunny_RestoreMatchesPythonReference()
    {
        string? dit = Env("SEEDVR2_DIT");
        string? vae = Env("SEEDVR2_VAE");
        string? emb = Env("SEEDVR2_EMB");
        string? refPath = Env("SEEDVR2_E2E_REF");
        string? framesDir = Env("SEEDVR2_FRAMES");
        if (dit is null || vae is null || emb is null || refPath is null || framesDir is null)
        {
            _output.WriteLine("SKIPPED: set SEEDVR2_DIT/VAE/EMB/E2E_REF/FRAMES.");
            return;
        }

        (Dictionary<string, Tensor> ditWeights, SafeTensorsLoader ditLoader) =
            SeedVr2CheckpointConverter.LoadAndConvert(dit);
        using SafeTensorsLoader _ = ditLoader;
        using SafeTensorsLoader vaeLoader = new();
        vaeLoader.Load(vae);
        Dictionary<string, Tensor> vaeWeights = vaeLoader.GetAllTensors();
        using SafeTensorsLoader embLoader = new();
        embLoader.Load(emb);
        Tensor posEmb = embLoader.GetTensor("pos_emb").CastTo(DType.F32);

        using SafeTensorsLoader refLoader = new();
        refLoader.Load(refPath);
        Tensor posteriorNoise = refLoader.GetTensor("posterior_noise");
        Tensor initNoise = refLoader.GetTensor("init_noise");
        Tensor expected = refLoader.GetTensor("output");   // (3,F,H,W) [-1,1]

        string backendSel = Env("SEEDVR2_E2E_BACKEND") ?? "cuda";
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(SeedVr2E2eParityTests).Assembly.Location)!, "Ptx");
        IBackend backend = backendSel == "cpu" || !Directory.Exists(ptxDir)
            ? new HartsyInference.Cpu.CpuBackend()
            : new HartsyInference.Cuda.CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);

        SeedVr2Config config = SeedVr2Config.Detect(ditWeights);
        SeedVr2Dit model = new(config);
        model.LoadWeights(ditWeights);
        SeedVr2VaeEncoder encoder = new(SeedVr2VaeConfig.Default);
        encoder.LoadWeights(vaeWeights);
        SeedVr2VaeDecoder decoder = new(SeedVr2VaeConfig.Default);
        decoder.LoadWeights(vaeWeights);

        List<byte[]> frames = new();
        int w = 0, h = 0;
        foreach (string f in Directory.GetFiles(framesDir, "*.png").OrderBy(x => x))
        {
            (byte[] rgb, int fw, int fh) = PngDecoder.DecodeFromFile(f);
            frames.Add(rgb);
            (w, h) = (fw, fh);
        }
        _output.WriteLine($"input: {frames.Count} frames {w}x{h}, backend {backendSel}");

        using SeedVr2RestorePipeline pipeline = new(backend, model, encoder, decoder, posEmb);
        pipeline.NoiseHook = (kind, chunk, shape) =>
        {
            Tensor src = kind == "posterior" ? posteriorNoise : initNoise;
            Assert.Equal(shape.ElementCount, src.Shape.ElementCount);
            Tensor copy = new Tensor(shape, DType.F32);
            src.AsSpan<float>().CopyTo(copy.AsSpan<float>());
            return copy;
        };

        // Gate area matches the Python side (SEEDVR2_AREA): f32 whole-clip VAE at 720p-area needs ~18+ GB
        // of activations, so the dtype-clean parity gate runs at a reduced area; full-res lands in Part F.
        long area = long.TryParse(Env("SEEDVR2_AREA"), out long a) ? a : 1280L * 720L;
        long t0 = Environment.TickCount64;
        (List<byte[]> restored, int outW, int outH) = pipeline.Restore(
            frames, w, h, new SeedVr2RestoreOptions { ClipFrames = frames.Count, TargetArea = area });
        long elapsed = Environment.TickCount64 - t0;
        _output.WriteLine($"restored {restored.Count} frames {outW}x{outH} in {elapsed} ms");

        int refF = (int)expected.Shape[1], refH = (int)expected.Shape[2], refW = (int)expected.Shape[3];
        Assert.Equal(refF, restored.Count);
        Assert.Equal((refH, refW), (outH, outW));

        ReadOnlySpan<float> exp = expected.AsSpan<float>();
        double ssimSum = 0, mseSum = 0;
        for (int f = 0; f < refF; f++)
        {
            byte[] refRgb = new byte[outW * outH * 3];
            for (int c = 0; c < 3; c++)
            {
                long plane = ((long)c * refF + f) * outH * outW;
                for (int i = 0; i < outH * outW; i++)
                {
                    float v = (Math.Clamp(exp[(int)(plane + i)], -1f, 1f) * 0.5f + 0.5f) * 255f;
                    refRgb[i * 3 + c] = (byte)Math.Clamp(MathF.Round(v), 0f, 255f);
                }
            }
            double ssim = Helpers.Ssim.Compute(restored[f], refRgb, outW, outH);
            double mse = 0;
            for (int i = 0; i < refRgb.Length; i++)
            {
                double d = restored[f][i] - refRgb[i];
                mse += d * d;
            }
            mse /= refRgb.Length;
            ssimSum += ssim;
            mseSum += mse;
            _output.WriteLine($"frame {f}: SSIM {ssim:f5}  PSNR {10 * Math.Log10(255.0 * 255.0 / Math.Max(mse, 1e-9)):f2} dB");
        }
        double meanSsim = ssimSum / refF;
        double meanPsnr = 10 * Math.Log10(255.0 * 255.0 / Math.Max(mseSum / refF, 1e-9));
        _output.WriteLine($"MEAN: SSIM {meanSsim:f5}  PSNR {meanPsnr:f2} dB");
        Assert.True(meanSsim >= 0.995, $"mean SSIM {meanSsim:f5} below 0.995 gate");
    }

    private static string? Env(string name)
    {
        string? v = Environment.GetEnvironmentVariable(name);
        return string.IsNullOrWhiteSpace(v) ? null : v;
    }
}
