using System.Globalization;
using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Encode-direction parity for <see cref="VaeEncoder"/> against a diffusers reference.
///
/// <para>This is the gate the img2img rollout depends on, and the reason it is a <i>reference</i> comparison rather
/// than a property test: <c>MODEL_STATUS_IMAGE.md</c> records a stride-2 asymmetric-padding bug that drove encode
/// correlation to 0.871, and notes "img2img always masked it". Strength-ordering, roundtrip-drift and mask-locality
/// assertions all compare img2img against itself, so every one of them passes with that bug live. Only a dumped
/// reference catches it.</para>
///
/// <para>The gate is keyed on <see cref="VaeConfig"/>, not on model family: one case here covers every family sharing
/// that VAE — <c>VaeConfig.Flux</c> alone backs chroma, flux1, z-image, lumina2, kandinsky5, f-lite, hidream, boogu and
/// omnigen2.</para>
///
/// Generate the references with:
/// <c>~/venvs/seedvr2/bin/python tests/python-reference/dump_vae_encode_reference.py --family all</c></summary>
[Trait("Category", "Integration")]
public sealed class VaeEncodeParityTests
{
    /// <summary>Encoder output is a deterministic conv stack, so agreement should be tight; the budget here is for
    /// F32-vs-F32 accumulation order only, well under the ~0.13 relL2 the padding bug produced.</summary>
    private const double MaxRelL2 = 0.02;

    /// <summary>Correlation is the sharper instrument for a spatial bug: the padding regression left relL2 plausible
    /// on interior pixels while dragging correlation to 0.871.</summary>
    private const double MinCorrelation = 0.999;

    private readonly ITestOutputHelper _output;

    public VaeEncodeParityTests(ITestOutputHelper output) => _output = output;

    /// <summary>Qwen-Image's 3D-causal VAE has its own encoder class and normalizes latents per channel rather than by
    /// a scalar pair, so it cannot ride the generic <see cref="VaeEncoder"/> case. This is the gate krea2 and anima
    /// depend on — they share <c>VaeConfig.QwenImage</c>.</summary>
    [Fact]
    public void QwenImageEncoder_MatchesDiffusersReference()
    {
        string refDir = Path.Combine(RepoRoot.Path, "tests/python-reference/vae_encode_reference/qwen-image");
        string weightPath = Path.Combine(TestPaths.ModelsDir, "VAE/QwenImage/qwen_image_vae.safetensors");
        if (!RealWeightGate.Require(_output.WriteLine, weightPath, Path.Combine(refDir, "index.json")))
        {
            return;
        }

        Reference reference = Reference.Load(refDir);
        using CpuBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(weightPath);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
        {
            DType dt = kvp.Value.DType;
            weights[kvp.Key] = dt == DType.F16 || dt == DType.BF16 ? kvp.Value.CastTo(DType.F32) : kvp.Value;
        }

        QwenImageVaeEncoder encoder = LoaderVaeUtils.TryBuildQwenEncoder(VaeConfig.QwenImage, weights, "parity:qwen-image")!;
        Assert.NotNull(encoder);
        using Tensor image = FromFile(Path.Combine(refDir, "image.bin"), reference.ImageShape);
        using Tensor actual = encoder.Encode(backend, image);
        using Tensor expected = FromFile(Path.Combine(refDir, "latent.bin"), reference.LatentShape);
        Assert.Equal(expected.Shape, actual.Shape);

        (double relL2, double correlation) = Compare(expected, actual);
        _output.WriteLine($"qwen-image: shape={actual.Shape} relL2={relL2:E3} corr={correlation:F6}");
        Assert.True(relL2 < MaxRelL2, $"qwen-image encode relL2 {relL2:E3} exceeds {MaxRelL2:E3}.");
        Assert.True(correlation > MinCorrelation, $"qwen-image encode correlation {correlation:F6} below {MinCorrelation}.");
    }

    [Theory]
    [InlineData("flux", "VAE/Flux/ae.safetensors")]
    [InlineData("flux2", "VAE/Flux/flux2-vae.safetensors")]
    [InlineData("sd3", "VAE/SD3/sd3.5_medium_vae_extracted.safetensors")]
    public void Encoder_MatchesDiffusersReference(string family, string relativeWeightPath)
    {
        string refDir = Path.Combine(RepoRoot.Path, "tests/python-reference/vae_encode_reference", family);
        string weightPath = Path.Combine(TestPaths.ModelsDir, relativeWeightPath);
        if (!RealWeightGate.Require(_output.WriteLine, weightPath, Path.Combine(refDir, "index.json")))
        {
            return;
        }

        Reference reference = Reference.Load(refDir);
        VaeConfig config = ConfigFor(family);

        // The scale/shift the reference was built with are transcribed from the model's published vae/config.json,
        // so this catches a VaeConfig that silently drifts from the checkpoint's real normalization.
        Assert.Equal(reference.ScalingFactor, config.ScalingFactor, 4);
        Assert.Equal(reference.ShiftFactor, config.ShiftFactor ?? 0.0f, 4);

        using CpuBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(weightPath);
        Dictionary<string, Tensor> weights = StageF32(loader);

        // Checked before the forward pass: a key holding the wrong tensor and a genuinely wrong network both
        // surface as a bad latent, and only this separates them.
        ReportWeightMismatches(reference, weights);

        VaeEncoder encoder = LoaderVaeUtils.BuildEncoder(config, weights, $"parity:{family}");
        using Tensor image = FromFile(Path.Combine(refDir, "image.bin"), reference.ImageShape);
        using Tensor actual = encoder.Encode(backend, image);

        using Tensor expected = FromFile(Path.Combine(refDir, "latent.bin"), reference.LatentShape);
        Assert.Equal(expected.Shape, actual.Shape);

        (double relL2, double correlation) = Compare(expected, actual);
        _output.WriteLine($"{family}: shape={actual.Shape} relL2={relL2:E3} corr={correlation:F6}");
        Assert.True(relL2 < MaxRelL2, $"{family} encode relL2 {relL2:E3} exceeds {MaxRelL2:E3}.");
        Assert.True(correlation > MinCorrelation, $"{family} encode correlation {correlation:F6} below {MinCorrelation}.");
    }

    /// <summary>Compares each staged encoder tensor's sum against the reference's, and fails naming the first few
    /// divergent keys — the fingerprint of an LDM-to-diffusers key mapping fault rather than bad network math.</summary>
    private void ReportWeightMismatches(Reference reference, IReadOnlyDictionary<string, Tensor> staged)
    {
        if (reference.EncoderWeightSums.Count == 0)
        {
            _output.WriteLine("reference carries no encoder_weight_sums; re-run the dump script to enable key diagnostics");
            return;
        }
        List<string> problems = [];
        foreach ((string key, double expected) in reference.EncoderWeightSums)
        {
            if (!staged.TryGetValue(key, out Tensor? tensor))
            {
                problems.Add($"{key}: MISSING on the C# side");
                continue;
            }
            double actual = 0.0;
            foreach (float v in tensor.AsSpan<float>())
            {
                actual += v;
            }
            // Sums over ~10^6 element tensors accumulate differently in float32 vs float64; scale the budget.
            double tolerance = Math.Max(1e-2, Math.Abs(expected) * 1e-4);
            if (Math.Abs(actual - expected) > tolerance)
            {
                problems.Add($"{key}: expected sum {expected.ToString("G8", CultureInfo.InvariantCulture)}, got {actual.ToString("G8", CultureInfo.InvariantCulture)}");
            }
        }
        if (problems.Count == 0)
        {
            _output.WriteLine($"encoder weight mapping matches the reference across {reference.EncoderWeightSums.Count} keys");
            return;
        }
        foreach (string problem in problems.Take(12))
        {
            _output.WriteLine($"  WEIGHT MISMATCH {problem}");
        }
        Assert.Fail($"{problems.Count} of {reference.EncoderWeightSums.Count} encoder weights do not match the reference "
            + $"— the key mapping is wrong, not the math. First: {problems[0]}");
    }

    private static VaeConfig ConfigFor(string family) => family switch
    {
        "flux" => VaeConfig.Flux,
        "flux2" => VaeConfig.Flux2,
        "sd3" => VaeConfig.Sd3,
        _ => throw new ArgumentOutOfRangeException(nameof(family), family, "No VaeConfig mapped for this reference."),
    };

    /// <summary>Stages every tensor as F32 under the diffusers key naming the encoder expects, mirroring what the
    /// recipes do at load time.</summary>
    private static Dictionary<string, Tensor> StageF32(SafeTensorsLoader loader)
    {
        Dictionary<string, Tensor> staged = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
        {
            // Extracted-from-checkpoint VAEs (SD3.5) keep the LDM `first_stage_model.` prefix.
            string key = kvp.Key.StartsWith("first_stage_model.", StringComparison.Ordinal)
                ? kvp.Key["first_stage_model.".Length..]
                : kvp.Key;
            string? diffusersKey = ModelAssets.CheckpointConverters.Utils.CheckpointConvertUtils.ConvertVaeKey(key);
            if (diffusersKey is null)
            {
                continue;
            }
            DType dt = kvp.Value.DType;
            staged[diffusersKey] = dt == DType.F16 || dt == DType.BF16 ? kvp.Value.CastTo(DType.F32) : kvp.Value;
        }
        return staged;
    }

    private static Tensor FromFile(string path, TensorShape shape)
    {
        byte[] bytes = File.ReadAllBytes(path);
        long expected = shape.ElementCount * sizeof(float);
        if (bytes.LongLength != expected)
        {
            throw new InvalidOperationException($"{path}: expected {expected} bytes for {shape}, got {bytes.LongLength}.");
        }
        Tensor tensor = new Tensor(shape, DType.F32);
        Span<float> dst = tensor.AsSpan<float>();
        for (int i = 0; i < dst.Length; i++)
        {
            dst[i] = BitConverter.ToSingle(bytes, i * sizeof(float));
        }
        return tensor;
    }

    private static (double RelL2, double Correlation) Compare(Tensor expected, Tensor actual)
    {
        ReadOnlySpan<float> e = expected.AsSpan<float>();
        ReadOnlySpan<float> a = actual.AsSpan<float>();
        double diffSq = 0.0, refSq = 0.0, sumE = 0.0, sumA = 0.0;
        for (int i = 0; i < e.Length; i++)
        {
            double d = a[i] - e[i];
            diffSq += d * d;
            refSq += (double)e[i] * e[i];
            sumE += e[i];
            sumA += a[i];
        }
        double meanE = sumE / e.Length, meanA = sumA / a.Length;
        double cov = 0.0, varE = 0.0, varA = 0.0;
        for (int i = 0; i < e.Length; i++)
        {
            double de = e[i] - meanE, da = a[i] - meanA;
            cov += de * da;
            varE += de * de;
            varA += da * da;
        }
        double relL2 = refSq > 0 ? Math.Sqrt(diffSq / refSq) : Math.Sqrt(diffSq);
        double corr = varE > 0 && varA > 0 ? cov / Math.Sqrt(varE * varA) : 0.0;
        return (relL2, corr);
    }

    /// <summary>The <c>index.json</c> the dump script writes alongside the tensors.</summary>
    private sealed record Reference(
        float ScalingFactor,
        float ShiftFactor,
        TensorShape ImageShape,
        TensorShape LatentShape,
        IReadOnlyDictionary<string, double> EncoderWeightSums)
    {
        public static Reference Load(string dir)
        {
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "index.json")));
            JsonElement root = doc.RootElement;
            Dictionary<string, double> sums = new Dictionary<string, double>(StringComparer.Ordinal);
            if (root.TryGetProperty("encoder_weight_sums", out JsonElement sumsElement))
            {
                foreach (JsonProperty prop in sumsElement.EnumerateObject())
                {
                    sums[prop.Name] = prop.Value.GetDouble();
                }
            }
            return new Reference(
                root.GetProperty("scaling_factor").GetSingle(),
                root.GetProperty("shift_factor").GetSingle(),
                ShapeOf(root.GetProperty("image")),
                ShapeOf(root.GetProperty("latent")),
                sums);
        }

        private static TensorShape ShapeOf(JsonElement stats)
        {
            List<long> dims = [];
            foreach (JsonElement dim in stats.GetProperty("shape").EnumerateArray())
            {
                dims.Add(dim.GetInt64());
            }
            return new TensorShape([.. dims]);
        }
    }
}
