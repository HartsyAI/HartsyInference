using System.Globalization;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models;
using HartsyInference.ThreeD.Models.Idol;
using HartsyInference.ThreeD.Preprocessing;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Idol's input on the CPU in float32 against idol-sdk at <c>956439f</c> (the dump of
/// <c>tests/python-reference/idol_reference/dump_idol_conditioning.py</c>): the picture preparation, then the two image streams of the
/// prepared picture. Needs the Comfy-Org/Idol repack (<c>HARTSY_IDOL_MODELS</c>, the folder that holds <c>idol/</c>) and the dump folder
/// (<c>HARTSY_IDOL_REF</c>). A missing input skips; with <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it fails.</summary>
[Trait("Category", "Integration")]
public sealed class IdolConditioningParityTests
{
    private const string ModelsEnv = "HARTSY_IDOL_MODELS";
    private const string ReferenceEnv = "HARTSY_IDOL_REF";
    private const string RequireEnv = "HARTSY_REQUIRE_REAL_WEIGHTS";

    // The engine's FLUX.2 VAE encoder differs from diffusers' float32 forward by a relative RMS of 4.2e-4 (peak-relative 7e-4) on this input; the
    // patchify and BatchNorm after it are exact. It was not bisected (the difference is far below the posterior noise Idol itself draws).
    private const double VaeTolerance = 1.5e-3;

    private readonly ITestOutputHelper _output;

    public IdolConditioningParityTests(ITestOutputHelper output) => _output = output;

    /// <summary>The preparation against idol-sdk's, on a picture that carries its own alpha: Lanczos resize, erode, crop and composite are all
    /// 8-bit integer arithmetic, so the prepared pixels should agree to a level.</summary>
    [Fact]
    public void Preparation_MatchesIdolSdk()
    {
        string? refDir = Environment.GetEnvironmentVariable(ReferenceEnv);
        string conditioning = Path.Combine(refDir ?? "", "cond", "conditioning.safetensors");
        if (!Available([conditioning]))
            return;

        using SafeTensorsLoader loader = new();
        loader.Load(conditioning);
        Dictionary<string, Tensor> r = loader.GetAllTensors();
        ReadOnlySpan<byte> rgba = r["input_rgba"].AsReadOnlySpan<byte>();
        int height = (int)r["input_rgba"].Shape[0], width = (int)r["input_rgba"].Shape[1];
        byte[] rgb = new byte[width * height * 3], alpha = new byte[width * height];
        for (int i = 0; i < width * height; i++)
        {
            rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2];
            alpha[i] = rgba[i * 4 + 3];
        }

        PreparedImage prepared = IdolImageProcessor.Process(new CpuBackend(), rgb, alpha, width, height, IdolFraming.Object, matter: null);

        ReadOnlySpan<byte> want = r["prepared"].AsReadOnlySpan<byte>();
        Assert.Equal(want.Length, prepared.Rgb.Length);
        int max = 0, different = 0;
        double sum = 0;
        for (int i = 0; i < want.Length; i++)
        {
            int d = Math.Abs(prepared.Rgb[i] - want[i]);
            max = Math.Max(max, d);
            sum += d;
            if (d != 0) different++;
        }

        _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"prepared 1024x1024 RGB: max |diff| {max} levels, {different} of {want.Length} values differ, mean |diff| {sum / want.Length:E2}"));
        Assert.True(max <= 2, $"the prepared pixels differ from idol-sdk's by up to {max} levels");
    }

    /// <summary>Both streams of the reference's prepared picture: DINOv3 ViT-H+ tokens and the FLUX.2 VAE tokens.</summary>
    [Fact]
    public void Streams_MatchIdolSdkOnTheCpu()
    {
        string? models = Environment.GetEnvironmentVariable(ModelsEnv);
        string? refDir = Environment.GetEnvironmentVariable(ReferenceEnv);
        string dino = Path.Combine(models ?? "", "idol", "clip_vision", "dino_v3_vit_h_bf16.safetensors");
        string vae = Path.Combine(models ?? "", "idol", "vae", "flux2-vae.safetensors");
        string conditioning = Path.Combine(refDir ?? "", "cond", "conditioning.safetensors");
        if (!Available([dino, vae, conditioning]))
            return;

        using SafeTensorsLoader loader = new();
        loader.Load(conditioning);
        Dictionary<string, Tensor> r = loader.GetAllTensors();
        byte[] prepared = r["prepared"].AsReadOnlySpan<byte>().ToArray();

        using Dinov3ImageConditioner dinoConditioner = Dinov3ImageConditioner.Load(dino, IdolImageProcessor.Size);
        using Flux2VaeComponents vaeComponents = Flux2VaeComponents.Load(vae, "idol");
        Assert.NotNull(vaeComponents.Encoder);
        CpuBackend backend = new();
        List<string> failures = [];

        using (Tensor feature1 = IdolConditioner.DinoTokens(backend, dinoConditioner, prepared))
            Report("feature1 (DINOv3 ViT-H+, final norm, LayerNorm)", feature1, r["feature1"], 1e-4, failures);
        using (Tensor feature2 = IdolConditioner.VaeTokens(backend, vaeComponents.Encoder!, vaeComponents.BnMean, vaeComponents.BnVar, prepared))
            Report("feature2 (FLUX.2 VAE mean, patchified, BN, 5 zero tokens)", feature2, r["feature2"], VaeTolerance, failures);
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    /// <summary>The VAE encoder on its own, to tell an encoder difference from the patchify and BatchNorm that follow it.</summary>
    [Fact]
    public void VaeLatent_MatchesDiffusersOnTheCpu()
    {
        string? models = Environment.GetEnvironmentVariable(ModelsEnv);
        string? refDir = Environment.GetEnvironmentVariable(ReferenceEnv);
        string vae = Path.Combine(models ?? "", "idol", "vae", "flux2-vae.safetensors");
        string conditioning = Path.Combine(refDir ?? "", "cond", "conditioning.safetensors");
        if (!Available([vae, conditioning]))
            return;

        using SafeTensorsLoader loader = new();
        loader.Load(conditioning);
        Dictionary<string, Tensor> r = loader.GetAllTensors();
        using Flux2VaeComponents components = Flux2VaeComponents.Load(vae, "idol");
        using Tensor pixels = new(r["pixels"].Shape, DType.F32);
        ReadOnlySpan<float> src = r["pixels"].AsReadOnlySpan<float>();
        Span<float> dst = pixels.AsSpan<float>();
        for (int i = 0; i < dst.Length; i++) dst[i] = src[i] * 2f - 1f;

        using Tensor latent = components.Encoder!.Encode(new CpuBackend(), pixels);

        List<string> failures = [];
        Report("VAE posterior mean [1, 32, 128, 128]", latent, r["vae_mean"], VaeTolerance, failures);
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    private bool Available(string[] paths)
    {
        string[] missing = paths.Where(p => !File.Exists(p)).ToArray();
        if (missing.Length == 0)
            return true;
        string message = $"required real-weight asset(s) missing: {string.Join(", ", missing)}";
        if (Environment.GetEnvironmentVariable(RequireEnv) == "1")
            throw new InvalidOperationException($"{RequireEnv}=1 but {message}.");
        _output.WriteLine($"SKIPPED: {message}");
        return false;
    }

    private void Report(string stage, Tensor got, Tensor want, double tolerance, List<string> failures)
    {
        ReadOnlySpan<float> g = got.AsReadOnlySpan<float>(), w = want.AsReadOnlySpan<float>();
        Assert.Equal(w.Length, g.Length);
        (double rel, double maxAbs) = ParityCompare.RelativeL2(g, w);
        double peak = 0;
        foreach (float v in w) peak = Math.Max(peak, Math.Abs(v));
        double peakRelative = maxAbs / peak;
        bool pass = peakRelative <= tolerance;
        _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{stage}: max abs {maxAbs:E3}, relRMS {rel:E2}, reference peak {peak:E3}, peak-relative {peakRelative:E2} vs {tolerance:E0}, {(pass ? "pass" : "FAIL")}"));
        if (!pass) failures.Add($"{stage} peak-relative error {peakRelative:E2} exceeds {tolerance:E0}");
    }
}
