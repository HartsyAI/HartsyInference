using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.ThreeD.Pipelines;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Hunyuan3D-2 DiT velocity and ShapeVAE occupancy against the fp32 CPU reference from <c>dump_shape_reference.py</c>.</summary>
/// <remarks>Integration lane: needs <c>HARTSY_PARITY_DIR</c> holding <c>hunyuan3d2_shape_fp32.npz</c> and the checkpoint under a Models folder
/// above the test binary or under <c>HARTSY_MODELS_DIR</c>. The DiT runs on the CPU backend in fp32 and takes minutes per forward. The
/// timestep comes from <see cref="Hunyuan3DFlowSchedule"/>, so a regression in the schedule or in the DiT's timestep contract fails here,
/// which per-component parity alone did not catch.</remarks>
[Trait("Category", "Integration")]
public sealed class Hunyuan3DShapeParityTests
{
    private const string Fixture = "hunyuan3d2_shape_fp32.npz";
    private const string ModelsRootVariable = "HARTSY_MODELS_DIR";
    private static readonly string CheckpointFile = Path.Combine("3D", "hunyuan3d2", "hunyuan3d-dit-v2-0", "model.fp16.safetensors");
    private readonly ITestOutputHelper _output;

    public Hunyuan3DShapeParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void DitVelocityAtTheScheduleTimestepMatchesTheReference()
    {
        if (!TryResolve(out string fixturePath, out string checkpoint, out string reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }

        NpzArchive fixture = NpzArchive.Open(fixturePath);
        int steps = fixture.Get("steps").ToInt32()[0];
        int[] saved = fixture.Get("saved_steps").ToInt32();
        int step = saved[saved.Length / 2];
        NpzArray condArray = fixture.Get("cond");
        float sigma = Hunyuan3DFlowSchedule.Sigmas(steps)[step];

        KnobStore.Set(EngineKnobs.DitF16, false);
        try
        {
            using CpuBackend backend = new();
            Hunyuan3DConfig config = Hunyuan3DConfig.Hunyuan3D2;
            Hunyuan3DDit dit = new(config);
            dit.LoadWeights(LoadWeights(checkpoint).Dit);
            Tensor cond = ToTensor(condArray.ToFloat32(), condArray.Shape);
            NpzArray latentArray = fixture.Get($"lat_in_step{step}");
            Tensor latent = ToTensor(latentArray.ToFloat32(), latentArray.Shape);

            Tensor velocity = dit.Forward(backend, latent, cond, Hunyuan3DFlowSchedule.DitTimestep(sigma));

            (double correlation, double relativeL2) = Compare(ToArray(velocity), fixture.Get($"v_cond_step{step}").ToFloat32());
            _output.WriteLine($"step {step}/{steps} sigma {sigma:F4}: correlation {correlation:F8}, relative L2 {relativeL2:E2}");
            Assert.True(correlation >= 0.99999 && relativeL2 <= 1e-3,
                $"velocity correlation {correlation}, relative L2 {relativeL2} at sigma {sigma}");
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.DitF16);
        }
    }

    [Fact]
    public void VaeOccupancyAtGridPointsMatchesTheReference()
    {
        if (!TryResolve(out string fixturePath, out string checkpoint, out string reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }

        NpzArchive fixture = NpzArchive.Open(fixturePath);
        Hunyuan3DConfig config = Hunyuan3DConfig.Hunyuan3D2;
        NpzArray grid = fixture.Get("grid_logits");
        int points = (int)grid.Shape[0];
        float[] reference = grid.ToFloat32();
        NpzArray latentArray = fixture.Get("latents_final");
        float[] latent = latentArray.ToFloat32();
        for (int i = 0; i < latent.Length; i++)
            latent[i] /= config.VaeScaleFactor;

        using CpuBackend backend = new();
        Hunyuan3DShapeVae vae = new(config);
        vae.LoadWeights(LoadWeights(checkpoint).ShapeVae);
        Tensor processed = vae.ProcessLatents(backend, ToTensor(latent, latentArray.Shape));
        (Tensor kNorm, Tensor v) kv = vae.PrepareKv(backend, processed);

        const int count = 2000;
        Random random = new(1);
        float[] coordinates = new float[count * 3];
        float[] expected = new float[count];
        for (int i = 0; i < count; i++)
        {
            int ix = random.Next(points), iy = random.Next(points), iz = random.Next(points);
            coordinates[i * 3] = Coordinate(ix, points);
            coordinates[i * 3 + 1] = Coordinate(iy, points);
            coordinates[i * 3 + 2] = Coordinate(iz, points);
            expected[i] = reference[(ix * points + iy) * points + iz];
        }

        Tensor occupancy = vae.DecodePoints(backend, kv, ToTensor(coordinates, [count, 3]), count);

        (double correlation, double relativeL2) = Compare(ToArray(occupancy), expected);
        _output.WriteLine($"occupancy at {count} grid points: correlation {correlation:F8}, relative L2 {relativeL2:E2}");
        Assert.True(correlation >= 0.99999 && relativeL2 <= 1e-3, $"occupancy correlation {correlation}, relative L2 {relativeL2}");
    }

    private static (double Correlation, double RelativeL2) Compare(float[] actual, float[] expected)
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

        return (covariance / Math.Sqrt(varianceA * varianceE), Math.Sqrt(squaredError / squaredExpected));
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

    private static bool TryResolve(out string fixturePath, out string checkpoint, out string reason)
    {
        string parityDir = Environment.GetEnvironmentVariable(ParityFixtures.ParityDirVariable) ?? string.Empty;
        fixturePath = Path.Combine(parityDir, Fixture);
        checkpoint = FindCheckpoint() ?? string.Empty;
        string? problem = parityDir.Length == 0 ? $"{ParityFixtures.ParityDirVariable} is not set"
            : !File.Exists(fixturePath) ? $"parity fixture {Fixture} not found in {parityDir}"
            : checkpoint.Length == 0 ? $"{CheckpointFile} not found under a Models folder above the test binary"
            : null;
        reason = problem ?? string.Empty;
        if (problem is not null && Environment.GetEnvironmentVariable(ParityFixtures.RequireWeightsVariable) == "1")
            throw new InvalidOperationException($"{ParityFixtures.RequireWeightsVariable}=1 but {problem}.");
        return problem is null;
    }

    private static string? FindCheckpoint()
    {
        string? modelsRoot = Environment.GetEnvironmentVariable(ModelsRootVariable);
        if (!string.IsNullOrWhiteSpace(modelsRoot) && File.Exists(Path.Combine(modelsRoot, CheckpointFile)))
            return Path.Combine(modelsRoot, CheckpointFile);
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, "Models", CheckpointFile);
            if (File.Exists(candidate))
                return candidate;
            dir = dir.Parent;
        }

        return null;
    }
}
