using System.Globalization;
using System.Security.Cryptography;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Dinov3;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Dinov3.Tests.Parity;

/// <summary>Real-weight parity of DINOv3 ViT-H+/16 (SwiGLU MLP) at 1024 px against Hugging Face's <c>DINOv3ViTModel</c> on the CPU
/// in float32, as dumped by <c>tests/python-reference/idol_reference/dump_idol_conditioning.py</c>. Needs the Comfy-Org/Idol
/// checkpoint (<c>HARTSY_DINOV3_H_WEIGHTS</c>, pinned by sha256) and the dump folder (<c>HARTSY_IDOL_REF</c>, holding
/// <c>cond/conditioning.safetensors</c>). A missing input skips the test; with <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it fails the run.</summary>
[Trait("Category", "Integration")]
public sealed class Dinov3VitHPlus16ParityTests
{
    private const string WeightsEnv = "HARTSY_DINOV3_H_WEIGHTS";
    private const string ReferenceEnv = "HARTSY_IDOL_REF";
    private const string RequireEnv = "HARTSY_REQUIRE_REAL_WEIGHTS";
    private const string PinnedSha256 = "a29ef35101a16966972a0d50732a6f3a608ff7cfffb2afa9bbe9007cb842cc53";
    private const int Resolution = 1024;

    // Peak-relative: max abs error over the stage's peak |reference| (the last blocks carry massive-activation channels in the
    // thousands). FP32 CPU against FP32 CPU; the starting points are the add-model values.
    private const double PreNormTolerance = 1e-4;
    private const double FinalTolerance = 1e-4;

    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    private readonly ITestOutputHelper _output;

    public Dinov3VitHPlus16ParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Tokens_MatchHuggingFaceOnTheCpu_At1024()
    {
        string? weightsPath = Environment.GetEnvironmentVariable(WeightsEnv);
        string? refDir = Environment.GetEnvironmentVariable(ReferenceEnv);
        string reference = Path.Combine(refDir ?? "", "cond", "conditioning.safetensors");
        string[] missing = new[] { weightsPath ?? WeightsEnv, reference }.Where(p => !File.Exists(p)).ToArray();
        if (missing.Length > 0)
        {
            string message = $"required real-weight asset(s) missing: {string.Join(", ", missing)}";
            if (Environment.GetEnvironmentVariable(RequireEnv) == "1")
                throw new InvalidOperationException($"{RequireEnv}=1 but {message}.");
            _output.WriteLine($"SKIPPED: {message}");
            return;
        }

        Assert.Equal(PinnedSha256, Sha256Of(weightsPath!), ignoreCase: true);
        using SafeTensorsLoader weightLoader = new();
        weightLoader.Load(weightsPath!);
        Dictionary<string, Tensor> weights = weightLoader.GetAllTensors();
        Dinov2Preset preset = Dinov2Preset.FromDinov3Weights(weights, Resolution);
        Assert.Equal(4101, preset.SequenceLength);
        Dinov3VisionEncoder encoder = new(preset);
        encoder.LoadWeights(weights);

        using SafeTensorsLoader refLoader = new();
        refLoader.Load(reference);
        Dictionary<string, Tensor> r = refLoader.GetAllTensors();
        using Tensor pixels = Normalize(r["pixels"]);

        CpuBackend backend = new();
        List<string> failures = [];
        using (Tensor preNorm = encoder.EncodeBlocks(backend, pixels, preset.NumLayers - 1))
            Report("last block (before the final norm)", preNorm, r["dino_prenorm"], PreNormTolerance, failures);
        using Tensor final = encoder.Encode(backend, pixels);
        Report("final norm (HF last_hidden_state)", final, r["dino_last"], FinalTolerance, failures);
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    /// <summary>The reference's own ImageNet normalisation of its [0, 1] pixels, <c>(x - mean) / std</c> in float32.</summary>
    private static Tensor Normalize(Tensor pixels)
    {
        Tensor o = new(pixels.Shape, DType.F32);
        ReadOnlySpan<float> src = pixels.AsReadOnlySpan<float>();
        Span<float> dst = o.AsSpan<float>();
        int plane = Resolution * Resolution;
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < plane; i++)
                dst[c * plane + i] = (src[c * plane + i] - Mean[c]) / Std[c];
        return o;
    }

    private void Report(string stage, Tensor got, Tensor want, double tolerance, List<string> failures)
    {
        ReadOnlySpan<float> w = want.AsReadOnlySpan<float>(), h = got.AsReadOnlySpan<float>();
        Assert.Equal(w.Length, h.Length);
        double maxAbs = 0, sumSq = 0, refSumSq = 0, peak = 0;
        for (int i = 0; i < h.Length; i++)
        {
            double d = h[i] - w[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(d));
            sumSq += d * d;
            refSumSq += (double)w[i] * w[i];
            peak = Math.Max(peak, Math.Abs((double)w[i]));
        }

        double rel = Math.Sqrt(sumSq / refSumSq);
        double peakRelative = maxAbs / peak;
        bool pass = peakRelative <= tolerance;
        _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{stage}: max abs {maxAbs:E3}, relRMS {rel:E2}, reference peak {peak:E3}, peak-relative {peakRelative:E2} vs {tolerance:E0}, {(pass ? "pass" : "FAIL")}"));
        if (!pass) failures.Add($"{stage} peak-relative error {peakRelative:E2} exceeds {tolerance:E0}");
    }

    private static string Sha256Of(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }
}
