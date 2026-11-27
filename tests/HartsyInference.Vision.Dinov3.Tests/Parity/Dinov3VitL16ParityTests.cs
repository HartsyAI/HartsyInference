using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Dinov3;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Dinov3.Tests.Parity;

/// <summary>Real-weight parity of DINOv3 ViT-L/16 at 448 px against the offline reference dump written by
/// <c>tools/parity/dump_dinov3_reference.py</c>. Needs the pinned checkpoint (<c>HARTSY_DINOV3_WEIGHTS</c>, or the default
/// path) and the dump directory (<c>HARTSY_DINOV3_REF</c>). When either is missing the test is skipped, not passed; with
/// <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> a missing input fails the run.</summary>
[Trait("Category", "Integration")]
public sealed class Dinov3VitL16ParityTests
{
    private const string WeightsEnv = "HARTSY_DINOV3_WEIGHTS";
    private const string ReferenceEnv = "HARTSY_DINOV3_REF";
    private const string RequireEnv = "HARTSY_REQUIRE_REAL_WEIGHTS";
    private const string DefaultWeights = @"C:\Users\AI Overlord\Models\Vision\dinov3-vitl16-pretrain-lvd1689m\model.safetensors";
    private const string PinnedSha256 = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179";
    private const int Resolution = 448;

    // Peak-relative tolerances: max abs error divided by the stage's peak |reference|. Block residuals carry
    // massive-activation channels in the hundreds to thousands, so an absolute bound would test the magnitude, not the math.
    // Starting points are the add-model values (GEMM 1e-5, attention 1e-4, full encoder 1e-3), FP32 CPU.
    private const double PixelTolerance = 1e-6;
    private const double EmbedTolerance = 1e-5;
    private const double BlockTolerance = 1e-4;
    private const double FinalTolerance = 1e-3;

    private readonly ITestOutputHelper _output;

    public Dinov3VitL16ParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void FinalFeatures_MatchOfflineReference_At448()
    {
        string weightsPath = Environment.GetEnvironmentVariable(WeightsEnv) ?? DefaultWeights;
        string? refDir = Environment.GetEnvironmentVariable(ReferenceEnv);
        string[] required = [weightsPath, Path.Combine(refDir ?? "", "meta.json")];
        string[] missing = required.Where(p => !File.Exists(p)).ToArray();
        if (missing.Length > 0)
        {
            string message = $"required real-weight asset(s) missing: {string.Join(", ", missing)}";
            if (Environment.GetEnvironmentVariable(RequireEnv) == "1")
                throw new InvalidOperationException($"{RequireEnv}=1 but {message}.");
            _output.WriteLine($"SKIPPED: {message}");
            return;
        }

        Assert.Equal(PinnedSha256, Sha256Of(weightsPath), ignoreCase: true);

        using SafeTensorsLoader loader = new();
        loader.Load(weightsPath);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        Dinov2Preset preset = Dinov2Preset.FromDinov3Weights(weights, Resolution);
        Assert.Equal(789, preset.SequenceLength);
        Dinov3VisionEncoder encoder = new(preset);
        encoder.LoadWeights(weights);

        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(refDir!, "meta.json")));
        _output.WriteLine($"reference torch {meta.RootElement.GetProperty("torch").GetString()}, "
            + $"official-vs-zero qkv mask max abs {meta.RootElement.GetProperty("official_mask_vs_zero_mask_max_abs").GetDouble():G6}");

        // Preprocessing: the same synthetic RGB the reference used, through the engine's ImageNet path.
        byte[] rgb = File.ReadAllBytes(Path.Combine(refDir!, "input_rgb_u8.bin"));
        using Tensor pixels = new Dinov2ImagePreprocessor(Resolution).Preprocess(rgb, Resolution, Resolution);
        List<string> failures = [];
        Report("pixels", pixels, Path.Combine(refDir!, "pixels_f32.bin"), PixelTolerance, failures);

        CpuBackend backend = new();
        using (Tensor embed = encoder.EncodeBlocks(backend, pixels, lastBlock: -1))
            Report("embed", embed, Path.Combine(refDir!, "embed_f32.bin"), EmbedTolerance, failures);
        using (Tensor block0 = encoder.EncodeBlocks(backend, pixels, lastBlock: 0))
            Report("block00", block0, Path.Combine(refDir!, "block00_f32.bin"), BlockTolerance, failures);
        using (Tensor block11 = encoder.EncodeBlocks(backend, pixels, lastBlock: 11))
            Report("block11", block11, Path.Combine(refDir!, "block11_f32.bin"), FinalTolerance, failures);
        using Tensor final = encoder.Encode(backend, pixels);
        Report("final", final, Path.Combine(refDir!, "final_f32.bin"), FinalTolerance, failures);
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    private void Report(string stage, Tensor got, string referencePath, double tolerance, List<string> failures)
    {
        ReadOnlySpan<float> want = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(referencePath));
        ReadOnlySpan<float> have = got.AsReadOnlySpan<float>();
        Assert.Equal(want.Length, have.Length);
        double maxAbs = 0, sumSq = 0, refSumSq = 0, peak = 0;
        for (int i = 0; i < have.Length; i++)
        {
            double d = have[i] - want[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(d));
            sumSq += d * d;
            refSumSq += (double)want[i] * want[i];
            peak = Math.Max(peak, Math.Abs((double)want[i]));
        }
        double rms = Math.Sqrt(sumSq / have.Length);
        double refRms = Math.Sqrt(refSumSq / have.Length);
        double peakRelative = maxAbs / peak;
        bool pass = peakRelative <= tolerance;
        _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{stage}: max abs {maxAbs:E3}, rms {rms:E3} (rel {rms / refRms:E2}), reference peak {peak:E3}, "
            + $"peak-relative {peakRelative:E2} vs {tolerance:E0}, {(pass ? "pass" : "FAIL")}"));
        if (!pass) failures.Add($"{stage} peak-relative error {peakRelative:E2} exceeds {tolerance:E0}");
    }

    private static string Sha256Of(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }
}
