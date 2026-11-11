using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.ThreeD.Pipelines;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Hunyuan3D-2 shape builds (base, Fast, Turbo, mini, mv) against the fp32 CPU reference from <c>dump_shape_reference.py</c>: the DiT velocity
/// at the schedule's timestep, the ShapeVAE occupancy and, for Turbo, the FlashVDM field.</summary>
/// <remarks>Integration lane: needs <c>HARTSY_PARITY_DIR</c> holding the build's <c>.npz</c> and its checkpoint folder (config.yaml and
/// model.fp16.safetensors) under a Models folder above the test binary or under <c>HARTSY_MODELS_DIR</c>. The DiT runs on the CPU backend
/// in fp32 and takes minutes per forward. The timestep and schedule come from <see cref="Hunyuan3DFlowSchedule"/> and the checkpoint's own config.</remarks>
[Trait("Category", "Integration")]
public sealed class Hunyuan3DShapeParityTests
{
    private const string ModelsRootVariable = "HARTSY_MODELS_DIR";
    private const string TurboVaeFolder = "hunyuan3d-vae-turbo";
    private readonly ITestOutputHelper _output;

    public Hunyuan3DShapeParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    public static TheoryData<string> Builds => new() { "base", "fast", "turbo", "mini", "mv" };

    [Theory]
    [MemberData(nameof(Builds))]
    public void DitVelocityAtTheScheduleTimestepMatchesTheReference(string build)
    {
        if (!TryResolve(build, out Build resolved, out string reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }

        NpzArchive fixture = NpzArchive.Open(resolved.Fixture);
        int steps = fixture.Get("steps").ToInt32()[0];
        int[] saved = fixture.Get("saved_steps").ToInt32();
        int step = saved[saved.Length / 2];
        Hunyuan3DConfig config = resolved.Config;
        float sigma = Hunyuan3DFlowSchedule.Sigmas(steps, config.Scheduler, config.PcmTimesteps)[step];
        NpzArray condArray = fixture.Get("cond");

        KnobStore.Set(EngineKnobs.DitF16, false);
        try
        {
            using CpuBackend backend = new();
            Hunyuan3DDit dit = new(config);
            dit.LoadWeights(LoadWeights(resolved.Checkpoint).Dit);
            Tensor cond = ToTensor(condArray.ToFloat32(), condArray.Shape);
            NpzArray latentArray = fixture.Get($"lat_in_step{step}");
            Tensor latent = ToTensor(latentArray.ToFloat32(), latentArray.Shape);

            Tensor velocity = dit.Forward(backend, latent, cond, Hunyuan3DFlowSchedule.DitTimestep(sigma),
                config.GuidanceEmbed ? config.GuidanceScale : null);

            (double correlation, double relativeL2) = Compare(ToArray(velocity), fixture.Get($"v_cond_step{step}").ToFloat32());
            _output.WriteLine($"{build} step {step}/{steps} sigma {sigma:F4}: correlation {correlation:F8}, relative L2 {relativeL2:E2}");
            Assert.True(correlation >= 0.99999 && relativeL2 <= 1e-3,
                $"{build} velocity correlation {correlation}, relative L2 {relativeL2} at sigma {sigma}");
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.DitF16);
        }
    }

    [Theory]
    [MemberData(nameof(Builds))]
    public void VaeOccupancyAtGridPointsMatchesTheReference(string build)
    {
        if (!TryResolve(build, out Build resolved, out string reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }

        NpzArchive fixture = NpzArchive.Open(resolved.Fixture);
        Hunyuan3DConfig config = resolved.Config;
        NpzArray grid = fixture.Get("grid_logits");
        int points = (int)grid.Shape[0];
        float[] reference = grid.ToFloat32();
        NpzArray latentArray = fixture.Get("latents_final");
        float[] latent = latentArray.ToFloat32();
        for (int i = 0; i < latent.Length; i++)
            latent[i] /= config.VaeScaleFactor;

        using CpuBackend backend = new();
        Hunyuan3DShapeVae vae = new(config);
        vae.LoadWeights(LoadWeights(resolved.Checkpoint).ShapeVae);
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
        _output.WriteLine($"{build} occupancy at {count} grid points: correlation {correlation:F8}, relative L2 {relativeL2:E2}");
        Assert.True(correlation >= 0.99999 && relativeL2 <= 1e-3, $"{build} occupancy correlation {correlation}, relative L2 {relativeL2}");
    }

    [Fact]
    public void FlashVdmFieldMatchesTheReference()
    {
        if (!TryResolve("turbo", out Build resolved, out string reason) || FindFolder(TurboVaeFolder) is not string vaeFolder)
        {
            _output.WriteLine($"SKIPPED: {(reason.Length > 0 ? reason : TurboVaeFolder + " not found")}");
            return;
        }

        NpzArchive fixture = NpzArchive.Open(resolved.Fixture);
        NpzArray grid = fixture.Get("flashvdm_grid");
        int n = (int)grid.Shape[0];
        float[] reference = grid.ToFloat32();   // x slowest, NaN where the hierarchy did not evaluate
        NpzArray latentArray = fixture.Get("latents_final");
        float[] latent = latentArray.ToFloat32();
        int octree = fixture.Get("octree_resolution").ToInt32()[0];

        using CpuBackend backend = new();
        Hunyuan3DConfig vaeConfig = Hunyuan3DConfigFile.ParseVae(File.ReadAllText(Path.Combine(vaeFolder, "config.yaml")), resolved.Config);
        Hunyuan3DShapeVae vae = new(vaeConfig);
        SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(vaeFolder, "model.fp16.safetensors"));
        Dictionary<string, Tensor> tensors = [];
        foreach ((string key, Tensor tensor) in loader.GetAllTensors())
            tensors[key] = tensor;
        vae.LoadWeights(tensors);
        for (int i = 0; i < latent.Length; i++)
            latent[i] /= vaeConfig.VaeScaleFactor;

        ScalarField3D field = vae.DecodeFlashVdm(backend, ToTensor(latent, latentArray.Shape), octree, bound: 1.01f);

        Assert.Equal(n, field.ResX);
        int onlyReference = 0, onlyCsharp = 0;
        List<float> actual = [], expected = [];
        for (int x = 0; x < n; x++)
        {
            for (int y = 0; y < n; y++)
            {
                for (int z = 0; z < n; z++)
                {
                    float r = reference[(x * n + y) * n + z], c = field.Values[x + n * (y + n * z)];
                    bool rNan = float.IsNaN(r), cNan = float.IsNaN(c);
                    if (rNan && cNan) continue;
                    if (rNan) onlyCsharp++;
                    else if (cNan) onlyReference++;
                    else
                    {
                        actual.Add(c);
                        expected.Add(r);
                    }
                }
            }
        }

        (double correlation, double relativeL2) = Compare([.. actual], [.. expected]);
        _output.WriteLine($"flashvdm {n}^3: shared {actual.Count}, only reference {onlyReference}, only C# {onlyCsharp}; correlation {correlation:F8}, relative L2 {relativeL2:E2}");
        Assert.True(onlyReference + onlyCsharp <= actual.Count / 100,
            $"the evaluated sets differ: {onlyReference} only in the reference, {onlyCsharp} only here, {actual.Count} shared");
        Assert.True(correlation >= 0.9999, $"flashvdm correlation {correlation}, relative L2 {relativeL2}");
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

    private static bool TryResolve(string build, out Build resolved, out string reason)
    {
        string fixtureName = build == "base" ? "hunyuan3d2_shape_fp32.npz" : $"hunyuan3d_{build}_shape_fp32.npz";
        string folderName = build == "base" ? Path.Combine("hunyuan3d2", "hunyuan3d-dit-v2-0") : $"hunyuan3d-{build}";
        string parityDir = Environment.GetEnvironmentVariable(ParityFixtures.ParityDirVariable) ?? string.Empty;
        string fixture = Path.Combine(parityDir, fixtureName);
        string? folder = FindFolder(folderName);
        string? problem = parityDir.Length == 0 ? $"{ParityFixtures.ParityDirVariable} is not set"
            : !File.Exists(fixture) ? $"parity fixture {fixtureName} not found in {parityDir}"
            : folder is null ? $"3D/{folderName} (config.yaml and model.fp16.safetensors) not found under a Models folder"
            : null;
        reason = problem ?? string.Empty;
        resolved = default;
        if (problem is not null)
        {
            if (Environment.GetEnvironmentVariable(ParityFixtures.RequireWeightsVariable) == "1")
                throw new InvalidOperationException($"{ParityFixtures.RequireWeightsVariable}=1 but {problem}.");
            return false;
        }

        resolved = new Build(fixture, Path.Combine(folder!, "model.fp16.safetensors"),
            Hunyuan3DConfigFile.Parse(File.ReadAllText(Path.Combine(folder!, "config.yaml"))));
        return true;
    }

    private static string? FindFolder(string folderName)
    {
        List<string> roots = [];
        if (Environment.GetEnvironmentVariable(ModelsRootVariable) is { Length: > 0 } explicitRoot)
            roots.Add(explicitRoot);
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
            roots.Add(Path.Combine(dir.FullName, "Models"));
        return roots.Select(root => Path.Combine(root, "3D", folderName))
            .FirstOrDefault(folder => File.Exists(Path.Combine(folder, "model.fp16.safetensors")) && File.Exists(Path.Combine(folder, "config.yaml")));
    }

    private readonly record struct Build(string Fixture, string Checkpoint, Hunyuan3DConfig Config);
}
