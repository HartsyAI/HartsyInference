using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;
using HartsyInference.Vision.Dinov2;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Integration lane: the Paint 2.1 UNet against upstream's own modules (on the CPU unless <c>HARTSY_PAINT21_BACKEND</c> says otherwise). The fixture comes from
/// <c>tests/python-reference/dump_paint21_unet.py</c> (inputs, the features the reference stream writes, the DINO projection, the UNet output for the full
/// and the unconditional branch, and a VAE decode). Set <c>HARTSY_PAINT21_UNET</c> (the pickle), <c>HARTSY_PAINT21_REF</c> (the fixture) and, for the VAE test,
/// <c>HARTSY_PAINT21_VAE</c>. Each stage logs the maximum absolute difference, the relative RMS error and the correlation.</summary>
[Trait("Category", "Integration")]
public sealed class HunyuanPaintUnet21ReferenceTests(ITestOutputHelper output)
{
    private static (double MaxAbs, double RelRms, double Correlation) Compare(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double max = 0, diff2 = 0, ref2 = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double a = actual[i], b = expected[i], d = a - b;
            max = Math.Max(max, Math.Abs(d));
            diff2 += d * d;
            ref2 += b * b;
            sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
        }
        double n = expected.Length, cov = sab / n - sa / n * (sb / n), va = saa / n - sa / n * (sa / n), vb = sbb / n - sb / n * (sb / n);
        return (max, Math.Sqrt(diff2 / Math.Max(ref2, 1e-30)), cov / Math.Sqrt(Math.Max(va * vb, 1e-30)));
    }

    private static string? BackendName => Environment.GetEnvironmentVariable("HARTSY_PAINT21_BACKEND") is { Length: > 0 } name ? name : null;

    private static bool OnGpu => BackendName is { } name && !string.Equals(name, "cpu", StringComparison.OrdinalIgnoreCase);

    /// <summary>Points the knob file at <c>HARTSY_PAINT21_SETTINGS</c> (numerics knobs such as <c>numerics.noTf32</c> and <c>numerics.sdpaNoTf32</c>) and reloads it, because
    /// the file loads once on the first knob read; disposing restores the previous path and reloads, so later tests in the process see their own settings. Without the
    /// variable it does nothing.</summary>
    private sealed class KnobScope : IDisposable
    {
        private readonly string? _previous = KnobFile.ExplicitPath;
        private readonly bool _active = Environment.GetEnvironmentVariable("HARTSY_PAINT21_SETTINGS") is { Length: > 0 };

        public KnobScope()
        {
            if (!_active)
                return;
            KnobFile.ExplicitPath = Environment.GetEnvironmentVariable("HARTSY_PAINT21_SETTINGS");
            KnobFile.Reload();
        }

        public void Dispose()
        {
            if (!_active)
                return;
            KnobFile.ExplicitPath = _previous;
            KnobFile.Reload();
        }
    }

    /// <summary>The backend of the lane: the CPU, or the one named by <c>HARTSY_PAINT21_BACKEND</c> (<c>cuda</c>, <c>vulkan</c>); <c>HARTSY_PAINT21_HP=1</c> turns on
    /// full-precision GEMMs. Create it inside a <see cref="KnobScope"/>, which the knobs are read through.</summary>
    private static IBackend Backend()
    {
        IBackend backend = BackendFactory.Create(BackendName ?? "cpu");
        if (Environment.GetEnvironmentVariable("HARTSY_PAINT21_HP") == "1")
            backend.HighPrecisionGemm = true;
        return backend;
    }

    // GPU stages that miss the loose bound; checked once at the end so that one bad stage does not hide the numbers of the later ones.
    private readonly List<string> _gpuFailures = [];

    private void AssertGpuBound() => Assert.True(_gpuFailures.Count == 0, "GPU stages outside the loose bound (relRMS 1e-2, correlation 0.999): " + string.Join("; ", _gpuFailures));

    private static unsafe float[] ToArray(Tensor t)
    {
        float[] values = new float[t.ElementCount];
        new ReadOnlySpan<float>((float*)Hunyuan3DShapeVae.F32(t).DataPointer, values.Length).CopyTo(values);
        return values;
    }

    private void Report(string stage, float[] actual, float[] expected)
    {
        (double max, double rel, double corr) = Compare(actual, expected);
        output.WriteLine($"{stage,-34} n={expected.Length,9}  max|diff|={max:E3}  relRMS={rel:E3}  corr={corr:F6}  refMax={expected.Max(Math.Abs):F3}");
        // A GPU run measures (TF32 is looser than the CPU tolerance) and is checked against a loose bound at the end.
        if (!OnGpu)
            Assert.True(rel < 5e-3 && corr > 0.9999, $"{stage}: relRMS {rel:E3}, correlation {corr:F6} are outside the tolerance (5e-3, 0.9999)");
        else if (rel >= 1e-2 || corr <= 0.999)
            _gpuFailures.Add($"{stage} relRMS {rel:E3} corr {corr:F6}");
    }

    [RealWeightsFact("HARTSY_PAINT21_UNET", "HARTSY_PAINT21_REF")]
    public unsafe void Unet_MatchesUpstream()
    {
        using SafeTensorsLoader fixtureLoader = new();
        fixtureLoader.Load(Environment.GetEnvironmentVariable("HARTSY_PAINT21_REF")!);
        Dictionary<string, Tensor> fixture = fixtureLoader.GetAllTensors();
        Tensor sample = fixture["sample"], normal = fixture["normal"], position = fixture["position"], maps = fixture["position_maps"], dino = fixture["dino"];
        int materials = (int)sample.Shape[1], views = (int)sample.Shape[2], latent = (int)sample.Shape[4], pixels = (int)maps.Shape[3];

        using PytorchPickleLoader pickle = new();
        pickle.Load(Environment.GetEnvironmentVariable("HARTSY_PAINT21_UNET")!);
        Dictionary<string, Tensor> weights = pickle.GetAllTensors();
        HunyuanPaintConfig config = HunyuanPaintLoader.Detect(weights);
        HunyuanPaintPromptTokens tokens = HunyuanPaintPromptTokens.FromWeights(weights, config, "unet");
        using HunyuanPaintUnet unet = new(config, generating: true), reference = new(config, generating: false);
        unet.LoadWeights(weights, "unet");
        reference.LoadWeights(weights, "unet_dual");
        using KnobScope knobs = new();
        using IBackend backend = Backend();
        System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();

        // Reference stream at timestep 0.
        HunyuanPaintBlockState write = new() { Write = true, Text = [tokens.Reference] };
        using Tensor referenceLatent = new(new TensorShape(1, 4, latent, latent), DType.F32);
        new ReadOnlySpan<float>((float*)fixture["ref_latents"].DataPointer, (int)referenceLatent.ElementCount).CopyTo(new Span<float>((float*)referenceLatent.DataPointer, (int)referenceLatent.ElementCount));
        foreach (Tensor t in reference.Forward(backend, [referenceLatent], 0f, [], write))
            t.Dispose();
        output.WriteLine($"reference stream forward: {watch.Elapsed.TotalSeconds:F1} s, {write.Embeds.Count} feature blocks");
        watch.Restart();
        double worstFeatureRel = 0;
        int compared = 0, fixtureBlocks = fixture.Keys.Count(k => k.StartsWith("ref.", StringComparison.Ordinal));
        foreach ((string name, Tensor features) in write.Embeds)
        {
            Assert.True(fixture.TryGetValue($"ref.{name}", out Tensor? expected), $"the fixture has no reference features for block {name}");
            float[] actualFeatures = ToArray(features), expectedFeatures = ToArray(expected!);
            if (compared++ < 4 || name.StartsWith("up_3_2", StringComparison.Ordinal) || name.StartsWith("mid", StringComparison.Ordinal))
                Report($"ref features {name}", actualFeatures, expectedFeatures);
            else
            {
                (_, double rel, double corr) = Compare(actualFeatures, expectedFeatures);
                worstFeatureRel = Math.Max(worstFeatureRel, rel);
                if (!OnGpu)
                    Assert.True(rel < 5e-3 && corr > 0.9999, $"reference features of block {name} are outside the tolerance");
                else if (rel >= 1e-2 || corr <= 0.999)
                    _gpuFailures.Add($"ref features {name} relRMS {rel:E3} corr {corr:F6}");
            }
        }
        output.WriteLine($"reference blocks compared: {compared} of {write.Embeds.Count}; fixture has {fixtureBlocks}; worst relRMS of the unlogged blocks {worstFeatureRel:E3}");
        Assert.Equal(fixtureBlocks, compared);

        // DINO projection.
        Tensor dinoCopy = new(dino.Shape, DType.F32);
        new ReadOnlySpan<float>((float*)dino.DataPointer, (int)dino.ElementCount).CopyTo(new Span<float>((float*)dinoCopy.DataPointer, (int)dino.ElementCount));
        using HunyuanPaintF32Weights f32 = new(weights);
        PaintDinoConditioner conditioner = new(new Dinov2VisionEncoder(Dinov2Preset.Giant), f32["unet.image_proj_model_dino.proj.weight"], f32["unet.image_proj_model_dino.proj.bias"],
            f32["unet.image_proj_model_dino.norm.weight"], f32["unet.image_proj_model_dino.norm.bias"]);
        using Tensor projected = conditioner.Project(backend, dinoCopy);
        Report("dino projection", ToArray(projected), ToArray(fixture["dino_proj"]));
        using Tensor zeroDino = conditioner.UnconditionalTokens(backend);

        // Rotation tables from the position maps (CHW float in [0, 1]). The fixture generator already rounds them to k / 255, so Math.Round(v * 255) below
        // recovers the exact bytes; change the generator and this round trip changes with it.
        byte[][] mapBytes = new byte[views][];
        float* mp = (float*)maps.DataPointer;
        for (int v = 0; v < views; v++)
        {
            mapBytes[v] = new byte[pixels * pixels * 3];
            for (int y = 0; y < pixels; y++)
                for (int x = 0; x < pixels; x++)
                    for (int c = 0; c < 3; c++)
                        mapBytes[v][(y * pixels + x) * 3 + c] = (byte)Math.Round(mp[(((long)v * 3 + c) * pixels + y) * pixels + x] * 255f);
        }
        Dictionary<int, (Tensor Cos, Tensor Sin)> rotation = HunyuanPaintPipeline.BuildRotation(mapBytes, views, pixels, latent, config.BlockOutChannels[0] / config.Heads[0]);

        Tensor[] Inputs()
        {
            Tensor[] inputs = new Tensor[materials];
            long plane = (long)latent * latent;
            for (int m = 0; m < materials; m++)
            {
                inputs[m] = new Tensor(new TensorShape(views, 12, latent, latent), DType.F32);
                float* o = (float*)inputs[m].DataPointer;
                for (int v = 0; v < views; v++)
                {
                    new ReadOnlySpan<float>((float*)sample.DataPointer + ((long)m * views + v) * 4 * plane, (int)(4 * plane)).CopyTo(new Span<float>(o + (long)v * 12 * plane, (int)(4 * plane)));
                    new ReadOnlySpan<float>((float*)normal.DataPointer + (long)v * 4 * plane, (int)(4 * plane)).CopyTo(new Span<float>(o + ((long)v * 12 + 4) * plane, (int)(4 * plane)));
                    new ReadOnlySpan<float>((float*)position.DataPointer + (long)v * 4 * plane, (int)(4 * plane)).CopyTo(new Span<float>(o + ((long)v * 12 + 8) * plane, (int)(4 * plane)));
                }
            }
            return inputs;
        }

        foreach ((string label, float scale, Tensor dinoTokens, string key) in new[] { ("unet output, full branch", 1f, projected, "out_full"), ("unet output, unconditional", 0f, zeroDino, "out_uncond") })
        {
            Tensor[] inputs = Inputs();
            watch.Restart();
            HunyuanPaintBlockState state = new() { Embeds = write.Embeds, ReferenceScale = scale, Text = tokens.PerMaterial, Dino = dinoTokens, Rotation = rotation };
            Tensor[] outputs = unet.Forward(backend, inputs, 500f, [], state);
            output.WriteLine($"{label}: forward {watch.Elapsed.TotalSeconds:F1} s");
            float[] actual = new float[materials * views * 4 * latent * latent];
            for (int m = 0; m < materials; m++)
                ToArray(outputs[m]).CopyTo(actual, m * views * 4 * latent * latent);
            Report(label, actual, ToArray(fixture[key]));
            foreach (Tensor t in inputs.Concat(outputs))
                t.Dispose();
        }
        foreach (Tensor t in write.Embeds.Values)
            t.Dispose();
        foreach ((Tensor cos, Tensor sin) in rotation.Values)
        {
            cos.Dispose();
            sin.Dispose();
        }
        AssertGpuBound();
    }

    /// <summary>The rotary voxel indices and the VAE decode against upstream. Set <c>HARTSY_PAINT21_AUX</c> (a fixture from <c>dump_paint21_unet.py --aux-only</c>)
    /// and <c>HARTSY_PAINT21_VAE</c> (the VAE pickle).</summary>
    [RealWeightsFact("HARTSY_PAINT21_AUX", "HARTSY_PAINT21_VAE")]
    public unsafe void VoxelIndicesAndVae_MatchUpstream()
    {
        using SafeTensorsLoader fixtureLoader = new();
        fixtureLoader.Load(Environment.GetEnvironmentVariable("HARTSY_PAINT21_AUX")!);
        Dictionary<string, Tensor> fixture = fixtureLoader.GetAllTensors();
        Tensor maps = fixture["position_maps"];
        int views = (int)maps.Shape[1], pixels = (int)maps.Shape[3];
        byte[] joined = new byte[views * pixels * pixels * 3];
        float* mp = (float*)maps.DataPointer;
        for (int v = 0; v < views; v++)
            for (int y = 0; y < pixels; y++)
                for (int x = 0; x < pixels; x++)
                    for (int c = 0; c < 3; c++)
                        joined[((v * pixels + y) * pixels + x) * 3 + c] = (byte)Math.Round(mp[(((long)v * 3 + c) * pixels + y) * pixels + x] * 255f);
        int latent = pixels / 8;
        for (int level = 0; level < 4; level++)
        {
            int grid = latent >> level, voxels = pixels >> level;
            int[] actual = HunyuanPaintRope.VoxelIndices(joined, views, pixels, grid, voxels);
            float* expected = (float*)fixture[$"voxel.{views * grid * grid}"].DataPointer;
            int differing = 0, maxDiff = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                int d = Math.Abs(actual[i] - (int)expected[i]);
                if (d != 0)
                    differing++;
                maxDiff = Math.Max(maxDiff, d);
            }
            output.WriteLine($"voxel indices grid {grid,3} resolution {voxels,4}: {differing} of {actual.Length} differ, max index difference {maxDiff}");
            Assert.Equal(0, differing);
        }

        using PytorchPickleLoader pickle = new();
        pickle.Load(Environment.GetEnvironmentVariable("HARTSY_PAINT21_VAE")!);
        using HunyuanPaintF32Weights weights = new(HunyuanPaintLoader.ModernizeVaeKeys(pickle.GetAllTensors()));
        HartsyInference.Diffusion.Models.Vae.VaeDecoder decoder = new(HartsyInference.Diffusion.Models.Vae.VaeConfig.Sd15);
        decoder.LoadWeights(weights);
        using KnobScope knobs = new();
        using IBackend backend = Backend();
        Tensor source = fixture["vae_latent"];
        using Tensor latents = new(source.Shape, DType.F32);
        new ReadOnlySpan<float>((float*)source.DataPointer, (int)source.ElementCount).CopyTo(new Span<float>((float*)latents.DataPointer, (int)source.ElementCount));
        System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();
        using Tensor decoded = decoder.Decode(backend, latents);
        output.WriteLine($"vae decode: {watch.Elapsed.TotalSeconds:F1} s");
        Report("vae decode (latent / 0.18215)", ToArray(decoded), ToArray(fixture["vae_decoded"]));
        AssertGpuBound();
    }

    /// <summary>The VAE encoder the pipeline runs on the reference picture and the normal and position maps, against upstream's posterior. Upstream draws <c>posterior.sample()</c> (mean plus
    /// standard deviation times noise); this repository's encoder returns the mean, so the test compares the mean and logs the size of the noise upstream adds. Set <c>HARTSY_PAINT21_AUX</c> to a fixture
    /// made with <c>dump_paint21_unet.py --aux-only --encode-image</c> and <c>HARTSY_PAINT21_VAE</c>.</summary>
    [RealWeightsFact("HARTSY_PAINT21_AUX", "HARTSY_PAINT21_VAE")]
    public unsafe void VaeEncode_MatchesUpstreamPosteriorMean_AndTheSamplingNoiseIsNegligible()
    {
        using SafeTensorsLoader fixtureLoader = new();
        fixtureLoader.Load(Environment.GetEnvironmentVariable("HARTSY_PAINT21_AUX")!);
        Dictionary<string, Tensor> fixture = fixtureLoader.GetAllTensors();
        Assert.True(fixture.ContainsKey("enc_mean"), "the fixture has no enc_mean: regenerate it with dump_paint21_unet.py --aux-only --encode-image <png>");

        using PytorchPickleLoader pickle = new();
        pickle.Load(Environment.GetEnvironmentVariable("HARTSY_PAINT21_VAE")!);
        using HunyuanPaintF32Weights weights = new(HunyuanPaintLoader.ModernizeVaeKeys(pickle.GetAllTensors()));
        HartsyInference.Diffusion.Models.Vae.VaeEncoder encoder = new(HartsyInference.Diffusion.Models.Vae.VaeConfig.Sd15);
        encoder.LoadWeights(weights);
        using KnobScope knobs = new();
        using IBackend backend = Backend();
        Tensor source = fixture["enc_image"];
        long count = source.ElementCount;
        using Tensor image = new(source.Shape, DType.F32);
        float* src = (float*)source.DataPointer, dst = (float*)image.DataPointer;
        for (long i = 0; i < count; i++)
            dst[i] = (src[i] - 0.5f) * 2f; // upstream's encode_images maps [0, 1] to [-1, 1] the same way; this test does not run the pipeline's own picture-to-tensor conversion
        System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();
        using Tensor encoded = encoder.Encode(backend, image);
        output.WriteLine($"vae encode: {watch.Elapsed.TotalSeconds:F1} s");

        float[] expectedMean = ToArray(fixture["enc_mean"]), std = ToArray(fixture["enc_std"]);
        Report("vae encode (posterior mean x 0.18215)", ToArray(encoded), expectedMean);
        // Per image (the picture, then the position map), because a white-background map is nearly constant: RMS(std) / RMS(mean) of each half of the batch.
        int half = std.Length / 2;
        for (int image2 = 0; image2 < 2; image2++)
        {
            double noise = Math.Sqrt(std.Skip(image2 * half).Take(half).Sum(v => (double)v * v) / half);
            double signal = Math.Sqrt(expectedMean.Skip(image2 * half).Take(half).Sum(v => (double)v * v) / half);
            output.WriteLine($"upstream's sampling noise, {(image2 == 0 ? "picture" : "position map")}: RMS {noise:E2} against a latent RMS of {signal:E2} (ratio {noise / signal:E2})");
            Assert.True(noise / signal < 0.01, $"the noise upstream adds to the {(image2 == 0 ? "picture" : "position map")} latents is not negligible against the mean this repository's encoder returns (a property of upstream's weights and the fixture)");
        }

        AssertGpuBound();
    }
}
