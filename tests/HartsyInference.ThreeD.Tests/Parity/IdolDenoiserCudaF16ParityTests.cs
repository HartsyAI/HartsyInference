using System.Globalization;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Idol;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>The shipped precision of Idol's denoiser (block matrices in F16, attention in F16, everything else F32, on CUDA) against idol-sdk's float32 dump at
/// the dense layout, so the F16 effect is a number and not only a look at a render. Same inputs as <see cref="IdolDenoiserParityTests"/>. Needs a CUDA card
/// (<c>HARTSY_IDOL_MODELS</c> and <c>HARTSY_IDOL_REF</c> as there); a missing input or card skips, and with <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> a missing input fails.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class IdolDenoiserCudaF16ParityTests
{
    private const string ModelsEnv = "HARTSY_IDOL_MODELS";
    private const string ReferenceEnv = "HARTSY_IDOL_REF";
    private const string RequireEnv = "HARTSY_REQUIRE_REAL_WEIGHTS";

    // idol-sdk runs its blocks in F16 too; against its float32 dump the engine's F16 forward lands inside 5e-3 of the peak and 1e-3 relative RMS (measured on the knight at t=731: 4.8e-3 and 9.0e-4).
    private const double PeakRelativeTolerance = 1e-2;
    private const double RelRmsTolerance = 2e-3;

    // Measured on the knight: logits 2.4e-4 to 5.5e-3 of the peak (relRMS up to 2.2e-3), features 2.9e-2 of the peak (relRMS 3.0e-3).
    private const double DecoderPeakRelativeTolerance = 5e-2;
    private const double DecoderRelRmsTolerance = 5e-3;

    private readonly ITestOutputHelper _output;

    public IdolDenoiserCudaF16ParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void DenseForwardInF16OnCuda_StaysCloseToIdolSdksFloat32()
    {
        string? models = Environment.GetEnvironmentVariable(ModelsEnv);
        string? refDir = Environment.GetEnvironmentVariable(ReferenceEnv);
        string checkpoint = Path.Combine(models ?? "", "idol", "diffusion_models", "idol_fp16.safetensors");
        string conditioning = Path.Combine(refDir ?? "", "cond", "conditioning.safetensors");
        string inputs = Path.Combine(refDir ?? "", "den", "denoiser_inputs.safetensors");
        string reference = Path.Combine(refDir ?? "", "den", "idol.safetensors");
        string[] missing = new[] { checkpoint, conditioning, inputs, reference }.Where(p => !File.Exists(p)).ToArray();
        if (missing.Length > 0)
        {
            string message = $"required real-weight asset(s) missing: {string.Join(", ", missing)}";
            if (Environment.GetEnvironmentVariable(RequireEnv) == "1")
                throw new InvalidOperationException($"{RequireEnv}=1 but {message}.");
            _output.WriteLine($"SKIPPED: {message}");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaBackend backend = new(0, PtxDir());
        using IdolWeights weights = IdolWeights.Load(checkpoint, linearDType: DType.F16);
        using IdolDenoiser model = new(weights);
        using SafeTensorsLoader condLoader = new(), inputLoader = new(), refLoader = new();
        condLoader.Load(conditioning);
        inputLoader.Load(inputs);
        refLoader.Load(reference);
        Dictionary<string, Tensor> cond = condLoader.GetAllTensors(), input = inputLoader.GetAllTensors(), want = refLoader.GetAllTensors();

        backend.PreloadWeights(model.EnumerateWeights());
        List<string> failures = [];
        try
        {
            using Tensor context = model.EmbedContext(backend, cond["feature1"], cond["feature2"]);
            (Tensor velocity, Tensor camera) = model.Forward(backend, input["latent"], input["camera"], 731f, context);
            using (velocity)
            using (camera)
            {
                backend.Sync();
                Report("v_latent", velocity, want["v_latent"], failures);
                Report("v_camera", camera, want["v_camera"], failures);
            }
        }
        finally
        {
            backend.Sync();
            backend.FreeWeights(model.EnumerateWeights());
        }

        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    /// <summary>The octree decoder (every level) and the Gaussian decoder in the shipped precision on CUDA, fed the reference's own latent, cell centres and anchors.</summary>
    [Fact]
    public void DecodersInF16OnCuda_StayCloseToIdolSdksFloat32()
    {
        string? models = Environment.GetEnvironmentVariable(ModelsEnv);
        string? refDir = Environment.GetEnvironmentVariable(ReferenceEnv);
        string checkpoint = Path.Combine(models ?? "", "idol", "vae", "idol_decoder_fp16.safetensors");
        string octreeDump = Path.Combine(refDir ?? "", "dec", "octree.safetensors");
        string gaussianDump = Path.Combine(refDir ?? "", "dec", "gaussians.safetensors");
        string[] missing = new[] { checkpoint, octreeDump, gaussianDump }.Where(p => !File.Exists(p)).ToArray();
        if (missing.Length > 0)
        {
            string message = $"required real-weight asset(s) missing: {string.Join(", ", missing)}";
            if (Environment.GetEnvironmentVariable(RequireEnv) == "1")
                throw new InvalidOperationException($"{RequireEnv}=1 but {message}.");
            _output.WriteLine($"SKIPPED: {message}");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaBackend backend = new(0, PtxDir());
        using IdolDecoderWeights weights = IdolDecoderWeights.Load(checkpoint, linearDType: DType.F16);
        using SafeTensorsLoader octreeLoader = new(), gaussianLoader = new();
        octreeLoader.Load(octreeDump);
        gaussianLoader.Load(gaussianDump);
        Dictionary<string, Tensor> octree = octreeLoader.GetAllTensors(), gaussian = gaussianLoader.GetAllTensors();
        Tensor latent = gaussian["latent"];
        List<string> failures = [];

        using (IdolOctreeDecoder decoder = new(weights))
        {
            for (int level = 1; octree.ContainsKey($"l{level}_logits"); level++)
            {
                float[] centres = octree[$"l{level}_x"].AsReadOnlySpan<float>().ToArray();
                int resolution = (int)octree[$"l{level}_level"].AsReadOnlySpan<long>()[0];
                float[] logits = decoder.Logits(backend, centres, resolution, latent);
                ReportArrays($"octree level {level} logits", logits, octree[$"l{level}_logits"].AsReadOnlySpan<float>().ToArray(), failures);
            }
        }

        float[] anchors = gaussian["anchors"].AsReadOnlySpan<float>().ToArray();
        using IdolGaussianDecoder gaussianDecoder = new(weights);
        using Tensor features = gaussianDecoder.Features(backend, anchors, latent);
        backend.Sync();
        ReportArrays("gaussian decoder features", features.AsReadOnlySpan<float>().ToArray(), gaussian["features"].AsReadOnlySpan<float>().ToArray(), failures);
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    private void ReportArrays(string stage, float[] got, float[] want, List<string> failures)
    {
        Assert.Equal(want.Length, got.Length);
        (double rel, double maxAbs) = ParityCompare.RelativeL2(got, want);
        double peak = 0;
        foreach (float v in want) peak = Math.Max(peak, Math.Abs(v));
        double peakRelative = maxAbs / peak;
        bool pass = peakRelative <= DecoderPeakRelativeTolerance && rel <= DecoderRelRmsTolerance;
        _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{stage} (CUDA, F16 blocks): max abs {maxAbs:E3}, relRMS {rel:E2}, reference peak {peak:E3}, peak-relative {peakRelative:E2} vs {DecoderPeakRelativeTolerance:E0}, relRMS limit {DecoderRelRmsTolerance:E0}, {(pass ? "pass" : "FAIL")}"));
        if (!pass) failures.Add($"{stage}: peak-relative {peakRelative:E2}, relRMS {rel:E2}");
    }

    private void Report(string stage, Tensor got, Tensor want, List<string> failures)
    {
        ReadOnlySpan<float> g = got.AsReadOnlySpan<float>(), w = want.AsReadOnlySpan<float>();
        Assert.Equal(w.Length, g.Length);
        (double rel, double maxAbs) = ParityCompare.RelativeL2(g, w);
        double peak = 0;
        foreach (float v in w) peak = Math.Max(peak, Math.Abs(v));
        double peakRelative = maxAbs / peak;
        bool pass = peakRelative <= PeakRelativeTolerance && rel <= RelRmsTolerance;
        _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{stage} (CUDA, F16 blocks): max abs {maxAbs:E3}, relRMS {rel:E2}, reference peak {peak:E3}, peak-relative {peakRelative:E2} vs {PeakRelativeTolerance:E0}, relRMS limit {RelRmsTolerance:E0}, {(pass ? "pass" : "FAIL")}"));
        if (!pass) failures.Add($"{stage}: peak-relative {peakRelative:E2}, relRMS {rel:E2}");
    }

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
                return candidate;
        }
        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx not found above the test output directory.");
    }
}
