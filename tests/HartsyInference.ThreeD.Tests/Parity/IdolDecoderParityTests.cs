using System.Globalization;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Idol;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Real-weight parity of Idol's two decoders on the CPU in float32 against idol-sdk at <c>956439f</c> (the dumps of
/// <c>tests/python-reference/idol_reference/dump_idol_decoders.py</c>). The anchor sampler draws from torch's generator, which the engine
/// cannot reproduce, so the octree decoder is checked on the per-level inputs the reference recorded and the Gaussian decoder on its
/// anchors. Needs the Comfy-Org/Idol repack (<c>HARTSY_IDOL_MODELS</c>, the folder that holds <c>idol/</c>) and the dump folder
/// (<c>HARTSY_IDOL_REF</c>). A missing input skips; with <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it fails.</summary>
[Trait("Category", "Integration")]
public sealed class IdolDecoderParityTests
{
    private const string ModelsEnv = "HARTSY_IDOL_MODELS";
    private const string ReferenceEnv = "HARTSY_IDOL_REF";
    private const string RequireEnv = "HARTSY_REQUIRE_REAL_WEIGHTS";

    // Peak-relative max error, FP32 CPU against FP32 CPU. The anchor embedding's angles reach 3e3 rad, where a float32 angle is good to 2e-4 rad.
    private const double LogitTolerance = 1e-3;
    private const double FeatureTolerance = 1e-3;
    private const double CloudTolerance = 1e-3;

    // The splat activations amplify a small feature error where a channel is large and negative (the log scale and the logit follow the raw
    // channel one for one there), so the derived arrays are judged by relative RMS, with the peak error reported.
    private const double CloudRmsTolerance = 3e-4;

    private readonly ITestOutputHelper _output;

    public IdolDecoderParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Decoders_MatchIdolSdkOnTheCpu()
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

        using IdolDecoderWeights weights = IdolDecoderWeights.Load(checkpoint);
        using SafeTensorsLoader octreeLoader = new(), gaussianLoader = new();
        octreeLoader.Load(octreeDump);
        gaussianLoader.Load(gaussianDump);
        Dictionary<string, Tensor> octree = octreeLoader.GetAllTensors(), gaussian = gaussianLoader.GetAllTensors();
        Tensor latent = gaussian["latent"];
        CpuBackend backend = new();
        List<string> failures = [];

        using (IdolOctreeDecoder decoder = new(weights))
        {
            for (int level = 1; octree.ContainsKey($"l{level}_logits"); level++)
            {
                float[] centres = octree[$"l{level}_x"].AsReadOnlySpan<float>().ToArray();
                // The reference passes the resolution 2^level as an int64 tensor.
                Assert.Equal(DType.I64, octree[$"l{level}_level"].DType);
                int resolution = (int)octree[$"l{level}_level"].AsReadOnlySpan<long>()[0];
                Assert.Equal(1 << level, resolution);
                float[] logits = decoder.Logits(backend, centres, resolution, latent);
                Report($"octree level {level} logits ({centres.Length / 3} cells)", logits, octree[$"l{level}_logits"].AsReadOnlySpan<float>().ToArray(), LogitTolerance, failures);
            }
        }

        float[] anchors = gaussian["anchors"].AsReadOnlySpan<float>().ToArray();
        using IdolGaussianDecoder gaussianDecoder = new(weights);
        using Tensor features = gaussianDecoder.Features(backend, anchors, latent);
        Report("gaussian decoder features", features.AsReadOnlySpan<float>().ToArray(), gaussian["features"].AsReadOnlySpan<float>().ToArray(), FeatureTolerance, failures);

        GaussianSplatCloud cloud = gaussianDecoder.ToCloud(anchors, features);
        Assert.Equal(262144, cloud.Count);
        Report("splat positions (y-up)", cloud.Positions, gaussian["ply_xyz"].AsReadOnlySpan<float>().ToArray(), CloudTolerance, failures);
        Report("splat colour", cloud.ShCoefficients, gaussian["ply_f_dc"].AsReadOnlySpan<float>().ToArray(), CloudRmsTolerance, failures, byRelRms: true);
        Report("splat log scale", cloud.Scales, gaussian["ply_scale"].AsReadOnlySpan<float>().ToArray(), CloudRmsTolerance, failures, byRelRms: true);
        Report("splat rotation", cloud.Rotations, gaussian["ply_rot"].AsReadOnlySpan<float>().ToArray(), CloudRmsTolerance, failures, byRelRms: true);
        Report("splat logit opacity", cloud.Opacities, gaussian["ply_opacity"].AsReadOnlySpan<float>().ToArray(), CloudRmsTolerance, failures, byRelRms: true);
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    /// <summary>The splat conversion on its own: the reference's own decoder output through <see cref="IdolGaussianDecoder.ToCloud"/> against
    /// the values idol-sdk writes to its PLY (y-up positions, log scales, logit opacities, turned rotations).</summary>
    [Fact]
    public void Cloud_FromTheReferenceFeatures_MatchesTheIdolSdkPly()
    {
        string? models = Environment.GetEnvironmentVariable(ModelsEnv);
        string? refDir = Environment.GetEnvironmentVariable(ReferenceEnv);
        string checkpoint = Path.Combine(models ?? "", "idol", "vae", "idol_decoder_fp16.safetensors");
        string gaussianDump = Path.Combine(refDir ?? "", "dec", "gaussians.safetensors");
        string[] missing = new[] { checkpoint, gaussianDump }.Where(p => !File.Exists(p)).ToArray();
        if (missing.Length > 0)
        {
            string message = $"required real-weight asset(s) missing: {string.Join(", ", missing)}";
            if (Environment.GetEnvironmentVariable(RequireEnv) == "1")
                throw new InvalidOperationException($"{RequireEnv}=1 but {message}.");
            _output.WriteLine($"SKIPPED: {message}");
            return;
        }

        using IdolDecoderWeights weights = IdolDecoderWeights.Load(checkpoint);
        using SafeTensorsLoader loader = new();
        loader.Load(gaussianDump);
        Dictionary<string, Tensor> gaussian = loader.GetAllTensors();
        using IdolGaussianDecoder decoder = new(weights);
        List<string> failures = [];

        GaussianSplatCloud cloud = decoder.ToCloud(gaussian["anchors"].AsReadOnlySpan<float>(), gaussian["features"]);

        Assert.Equal(262144, cloud.Count);
        Report("splat positions (y-up)", cloud.Positions, gaussian["ply_xyz"].AsReadOnlySpan<float>().ToArray(), 1e-5, failures);
        Report("splat colour", cloud.ShCoefficients, gaussian["ply_f_dc"].AsReadOnlySpan<float>().ToArray(), 1e-6, failures);
        Report("splat log scale", cloud.Scales, gaussian["ply_scale"].AsReadOnlySpan<float>().ToArray(), 1e-4, failures);
        Report("splat rotation", cloud.Rotations, gaussian["ply_rot"].AsReadOnlySpan<float>().ToArray(), 1e-5, failures);
        Report("splat logit opacity", cloud.Opacities, gaussian["ply_opacity"].AsReadOnlySpan<float>().ToArray(), 1e-4, failures);
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    private void Report(string stage, float[] got, float[] want, double tolerance, List<string> failures, bool byRelRms = false)
    {
        Assert.Equal(want.Length, got.Length);
        (double rel, double maxAbs) = ParityCompare.RelativeL2(got, want);
        double peak = 0;
        foreach (float v in want) peak = Math.Max(peak, Math.Abs(v));
        double peakRelative = maxAbs / peak;
        bool pass = (byRelRms ? rel : peakRelative) <= tolerance;
        _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{stage}: max abs {maxAbs:E3}, relRMS {rel:E2}, reference peak {peak:E3}, peak-relative {peakRelative:E2}, gate {(byRelRms ? "relRMS" : "peak-relative")} {tolerance:E0}, {(pass ? "pass" : "FAIL")}"));
        if (!pass) failures.Add($"{stage} {(byRelRms ? "relative RMS" : "peak-relative")} error exceeds {tolerance:E0}");
    }
}
