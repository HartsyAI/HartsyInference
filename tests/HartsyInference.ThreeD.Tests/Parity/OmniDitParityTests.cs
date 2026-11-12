using System.Diagnostics;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.ThreeD.Models.Hunyuan3D.Omni;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>The Hunyuan3D-Omni DiT, loaded from the released <c>model/pytorch_model.bin</c> through <see cref="OmniCheckpoint"/>, against upstream's own <c>HunYuanDiTPlain</c> in float32 on the CPU
/// (<c>tests/python-reference/dump_omni_dit.py</c>), with a context of 1370 DINO-like tokens, 7 control tokens and 10 signal tokens.</summary>
/// <remarks>Integration lane: needs <c>HARTSY_PARITY_DIR</c> holding <c>omni_dit_fp32.npz</c> and the Omni folder under <c>HARTSY_MODELS_DIR</c> (or a Models folder above the test binary) as
/// <c>3D/hunyuan3d-omni</c>. The float32 model needs about 12.5 GB of memory and the forward takes tens of minutes on the CPU backend.</remarks>
[Trait("Category", "Integration")]
public sealed class OmniDitParityTests(ITestOutputHelper output)
{
    private const string Fixture = "omni_dit_fp32.npz";

    [Fact]
    public void DitVelocityAndBlockOutputsMatchTheReference()
    {
        if (!TryResolve(out string fixturePath, out string folder, out string reason))
        {
            output.WriteLine($"SKIPPED: {reason}");
            return;
        }

        NpzArchive fixture = NpzArchive.Open(fixturePath);
        float[] sigmas = fixture.Get("sigmas").ToFloat32();
        NpzArray condArray = fixture.Get("cond"), noiseArray = fixture.Get("noise");

        Stopwatch watch = Stopwatch.StartNew();
        using OmniCheckpoint checkpoint = OmniCheckpoint.Load(folder, halfWeights: false);
        output.WriteLine($"read {checkpoint.Dit.Count} DiT tensors in {watch.Elapsed.TotalSeconds:F0} s");
        Hunyuan3DConfig config = checkpoint.Config;
        Assert.Equal(752, checkpoint.Dit.Count);

        using CpuBackend backend = new();
        Hunyuan3DPlainDit dit = new(config);
        dit.LoadWeights(checkpoint.Dit);
        dit.BlockProbe = (layer, hidden) =>
        {
            if (fixture.Has($"block{layer}"))
                Check($"block {layer} output", ToArray(hidden), fixture.Get($"block{layer}").ToFloat32(), minCorrelation: 0.99999, maxRelativeL2: 1e-3);
        };
        using (Tensor probeLatent = ToTensor(noiseArray.ToFloat32(), noiseArray.Shape), embedded = dit.Embed(backend, probeLatent, sigmas[0]))
        {
            float[] all = ToArray(embedded);
            Check("t_token", all[..config.Width], fixture.Get("t_token").ToFloat32(), minCorrelation: 0.99999, maxRelativeL2: 1e-3);
            Check("x_embedded", all[config.Width..], fixture.Get("x_embedded").ToFloat32(), minCorrelation: 0.99999, maxRelativeL2: 1e-3);
        }

        watch.Restart();
        using Tensor noise = ToTensor(noiseArray.ToFloat32(), noiseArray.Shape), cond = ToTensor(condArray.ToFloat32(), condArray.Shape);
        using Tensor velocity = dit.Forward(backend, noise, cond, sigmas[0]);
        output.WriteLine($"forward: {watch.Elapsed.TotalSeconds:F0} s for a context of {cond.Shape[1]} tokens");
        Check($"velocity, sigma {sigmas[0]:F2}", ToArray(velocity), fixture.Get("velocity_cond_0").ToFloat32(), minCorrelation: 0.99999, maxRelativeL2: 1e-3);
    }

    private void Check(string what, float[] actual, float[] expected, double minCorrelation, double maxRelativeL2)
    {
        Assert.Equal(expected.Length, actual.Length);
        double meanA = actual.Average(), meanE = expected.Average();
        double covariance = 0, varianceA = 0, varianceE = 0, squaredError = 0, squaredExpected = 0, worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double da = actual[i] - meanA, de = expected[i] - meanE, diff = actual[i] - expected[i];
            covariance += da * de;
            varianceA += da * da;
            varianceE += de * de;
            squaredError += diff * diff;
            squaredExpected += (double)expected[i] * expected[i];
            worst = Math.Max(worst, Math.Abs(diff));
        }

        double correlation = covariance / Math.Sqrt(varianceA * varianceE), relativeL2 = Math.Sqrt(squaredError / squaredExpected);
        output.WriteLine($"{what}: correlation {correlation:F8}, relative L2 {relativeL2:E2}, largest absolute difference {worst:E2}");
        Assert.True(correlation >= minCorrelation && relativeL2 <= maxRelativeL2, $"{what}: correlation {correlation}, relative L2 {relativeL2}");
    }

    private static bool TryResolve(out string fixturePath, out string folder, out string reason)
    {
        string parityDir = Environment.GetEnvironmentVariable(ParityFixtures.ParityDirVariable) ?? string.Empty;
        fixturePath = Path.Combine(parityDir, Fixture);
        folder = FindFolder() ?? string.Empty;
        string? problem = parityDir.Length == 0 ? $"{ParityFixtures.ParityDirVariable} is not set"
            : !File.Exists(fixturePath) ? $"parity fixture {Fixture} not found in {parityDir}"
            : folder.Length == 0 ? "3D/hunyuan3d-omni (config.json, model, vae and cond_encoder) not found under a Models folder"
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
        return roots.Select(root => Path.Combine(root, "3D", "hunyuan3d-omni"))
            .FirstOrDefault(folder => File.Exists(Path.Combine(folder, "model", "pytorch_model.bin")) && File.Exists(Path.Combine(folder, "vae", "pytorch_model.bin"))
                && File.Exists(Path.Combine(folder, "cond_encoder", "pytorch_model.bin")) && File.Exists(Path.Combine(folder, "config.json")));
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
}
