using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Hunyuan3D-2.1 DiT velocity and bundled-VAE occupancy against the fp32 CPU reference from <c>dump_shape21_reference.py</c>.</summary>
/// <remarks>Integration lane: needs <c>HARTSY_PARITY_DIR</c> holding <c>hunyuan3d_21_shape_fp32.npz</c> and <c>3D/hunyuan3d-21/dit</c>
/// (<c>model.fp16.safetensors</c> repacked from the upstream pickle, and <c>config.yaml</c>) under a Models folder above the test binary or
/// under <c>HARTSY_MODELS_DIR</c>. The fp32 model needs about 12 GB of memory and each forward takes minutes on the CPU backend.</remarks>
[Trait("Category", "Integration")]
public sealed class Hunyuan3D21ShapeParityTests
{
    private const string Fixture = "hunyuan3d_21_shape_fp32.npz";
    private readonly ITestOutputHelper _output;

    public Hunyuan3D21ShapeParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void DitVelocityMatchesTheReferenceForImageAndZeroTokens()
    {
        if (!TryResolve(out string fixturePath, out string folder, out string reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }

        NpzArchive fixture = NpzArchive.Open(fixturePath);
        Hunyuan3DConfig config = Hunyuan3DConfigFile.Parse(File.ReadAllText(Path.Combine(folder, "config.yaml")));
        float[] sigmas = fixture.Get("sigmas").ToFloat32();
        NpzArray condArray = fixture.Get("cond"), noiseArray = fixture.Get("noise");

        using CpuBackend backend = new();
        Hunyuan3DPlainDit dit = new(config);
        dit.LoadWeights(LoadWeights(Path.Combine(folder, "model.fp16.safetensors")).Dit);
        float[] condValues = condArray.ToFloat32();
        for (int n = 0; n < sigmas.Length; n++)
        {
            Tensor velocity = dit.Forward(backend, ToTensor(noiseArray.ToFloat32(), noiseArray.Shape), ToTensor(condValues, condArray.Shape), sigmas[n]);
            Check($"image tokens, sigma {sigmas[n]:F2}", ToArray(velocity), fixture.Get($"velocity_cond_{n}").ToFloat32());
        }

        Tensor zero = ToTensor(new float[condValues.Length], condArray.Shape);
        Tensor unconditional = dit.Forward(backend, ToTensor(noiseArray.ToFloat32(), noiseArray.Shape), zero, sigmas[0]);
        Check($"zero tokens, sigma {sigmas[0]:F2}", ToArray(unconditional), fixture.Get("velocity_uncond_0").ToFloat32());
    }

    [Fact]
    public void VaeOccupancyAtGridPointsMatchesTheReference()
    {
        if (!TryResolve(out string fixturePath, out string folder, out string reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }

        NpzArchive fixture = NpzArchive.Open(fixturePath);
        Hunyuan3DConfig config = Hunyuan3DConfigFile.Parse(File.ReadAllText(Path.Combine(folder, "config.yaml")));
        NpzArray grid = fixture.Get("vae_grid"), latentArray = fixture.Get("vae_latents");
        int points = (int)grid.Shape[0];
        float[] reference = grid.ToFloat32();
        float[] latent = latentArray.ToFloat32();
        for (int i = 0; i < latent.Length; i++)
            latent[i] /= config.VaeScaleFactor;

        using CpuBackend backend = new();
        Hunyuan3DShapeVae vae = new(config);
        vae.LoadWeights(LoadWeights(Path.Combine(folder, "model.fp16.safetensors")).ShapeVae);
        Tensor processed = vae.ProcessLatents(backend, ToTensor(latent, latentArray.Shape));
        (Tensor kNorm, Tensor v) kv = vae.PrepareKv(backend, processed);

        const int count = 2000;
        Random random = new(1);
        float[] coordinates = new float[count * 3], expected = new float[count];
        for (int i = 0; i < count; i++)
        {
            int ix = random.Next(points), iy = random.Next(points), iz = random.Next(points);
            coordinates[i * 3] = Coordinate(ix, points);
            coordinates[i * 3 + 1] = Coordinate(iy, points);
            coordinates[i * 3 + 2] = Coordinate(iz, points);
            expected[i] = reference[(ix * points + iy) * points + iz];
        }

        Tensor occupancy = vae.DecodePoints(backend, kv, ToTensor(coordinates, [count, 3]), count);
        Check($"occupancy at {count} grid points", ToArray(occupancy), expected);
    }

    private void Check(string what, float[] actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double meanA = actual.Average(), meanE = expected.Average();
        double covariance = 0, varianceA = 0, varianceE = 0, squaredError = 0, squaredExpected = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double da = actual[i] - meanA, de = expected[i] - meanE, diff = actual[i] - expected[i];
            covariance += da * de;
            varianceA += da * da;
            varianceE += de * de;
            squaredError += diff * diff;
            squaredExpected += (double)expected[i] * expected[i];
        }

        double correlation = covariance / Math.Sqrt(varianceA * varianceE), relativeL2 = Math.Sqrt(squaredError / squaredExpected);
        _output.WriteLine($"{what}: correlation {correlation:F8}, relative L2 {relativeL2:E2}");
        Assert.True(correlation >= 0.99999 && relativeL2 <= 1e-3, $"{what}: correlation {correlation}, relative L2 {relativeL2}");
    }

    private static float Coordinate(int index, int points) => -1.01f + 2.02f * index / (points - 1);

    private static Hunyuan3DCheckpointConverter.ConvertedWeights LoadWeights(string checkpoint)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpoint);
        Dictionary<string, Tensor> all = [];
        foreach ((string key, Tensor tensor) in loader.GetAllTensors())
            all[key] = tensor;
        return Hunyuan3DCheckpointConverter.Convert(all);
    }

    private static unsafe Tensor ToTensor(float[] values, long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        values.AsSpan().CopyTo(new Span<float>((float*)tensor.DataPointer, values.Length));
        return tensor;
    }

    private static unsafe float[] ToArray(Tensor tensor)
    {
        Tensor f32 = TensorCasts.EnsureF32(tensor);
        float[] values = new float[tensor.ElementCount];
        new ReadOnlySpan<float>((float*)f32.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    private static bool TryResolve(out string fixturePath, out string folder, out string reason)
    {
        string parityDir = Environment.GetEnvironmentVariable(ParityFixtures.ParityDirVariable) ?? string.Empty;
        fixturePath = Path.Combine(parityDir, Fixture);
        folder = FindFolder() ?? string.Empty;
        string? problem = parityDir.Length == 0 ? $"{ParityFixtures.ParityDirVariable} is not set"
            : !File.Exists(fixturePath) ? $"parity fixture {Fixture} not found in {parityDir}"
            : folder.Length == 0 ? "3D/hunyuan3d-21/dit (config.yaml and model.fp16.safetensors) not found under a Models folder"
            : null;
        reason = problem ?? string.Empty;
        if (problem is not null && Environment.GetEnvironmentVariable(ParityFixtures.RequireWeightsVariable) == "1")
            throw new InvalidOperationException($"{ParityFixtures.RequireWeightsVariable}=1 but {problem}.");
        return problem is null;
    }

    private static string? FindFolder()
    {
        List<string> roots = [];
        if (Environment.GetEnvironmentVariable("HARTSY_MODELS_DIR") is { Length: > 0 } explicitRoot)
            roots.Add(explicitRoot);
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
            roots.Add(Path.Combine(dir.FullName, "Models"));
        return roots.Select(root => Path.Combine(root, "3D", "hunyuan3d-21", "dit"))
            .FirstOrDefault(folder => File.Exists(Path.Combine(folder, "model.fp16.safetensors")) && File.Exists(Path.Combine(folder, "config.yaml")));
    }
}
