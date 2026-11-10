using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>SANA-WM schedule against diffusers 0.37.0 fixtures generated on CPU
/// (<c>Fixtures/generate_sana_wm_schedule_fixtures.py</c>). The builder follows diffusers' operation order, so every
/// float32 sigma and timestep is expected to match bit for bit.</summary>
public sealed class SanaWmScheduleTests
{
    [Theory]
    [InlineData("sana_wm_schedule_steps60_shift98.json")]
    [InlineData("sana_wm_schedule_steps60_shift80.json")]
    [InlineData("sana_wm_schedule_steps30_shift98.json")]
    [InlineData("sana_wm_schedule_steps1_shift98.json")]
    [InlineData("sana_wm_schedule_steps2_shift98.json")]
    public void Build_MatchesDiffusersFloat32Schedule_Bitwise(string fixture)
    {
        using JsonDocument doc = LoadFixture(fixture);
        JsonElement root = doc.RootElement;
        int steps = root.GetProperty("steps").GetInt32();
        double shift = root.GetProperty("shift").GetDouble();

        (float[] sigmas, float[] timesteps) = SanaWmSchedule.Build(steps, shift);

        AssertSameBits(ReadFloats(root, "sigmas_float32"), sigmas, "sigmas");
        AssertSameBits(ReadFloats(root, "timesteps_float32"), timesteps, "timesteps");
    }

    [Theory]
    [InlineData("sana_wm_schedule_steps60_shift98.json")]
    [InlineData("sana_wm_schedule_steps60_shift80.json")]
    [InlineData("sana_wm_schedule_steps30_shift98.json")]
    [InlineData("sana_wm_schedule_steps1_shift98.json")]
    [InlineData("sana_wm_schedule_steps2_shift98.json")]
    public void Fixture_Float64Sigmas_RoundToFloat32Sigmas(string fixture)
    {
        using JsonDocument doc = LoadFixture(fixture);
        float[] sigmas64 = ReadFloats(doc.RootElement, "sigmas_float64");
        float[] sigmas32 = ReadFloats(doc.RootElement, "sigmas_float32");

        Assert.Equal(sigmas32.Length, sigmas64.Length);
        for (int i = 0; i < sigmas64.Length; i++)
        {
            Assert.Equal(BitConverter.SingleToInt32Bits(sigmas32[i]), BitConverter.SingleToInt32Bits(sigmas64[i]));
        }
    }

    [Theory]
    [InlineData("sana_wm_schedule_steps60_shift98.json")]
    [InlineData("sana_wm_schedule_steps60_shift80.json")]
    [InlineData("sana_wm_schedule_steps30_shift98.json")]
    [InlineData("sana_wm_schedule_steps1_shift98.json")]
    [InlineData("sana_wm_schedule_steps2_shift98.json")]
    public void SetSigmas_WithBuiltSchedule_ReportsFixtureSigmasAndTimesteps(string fixture)
    {
        using JsonDocument doc = LoadFixture(fixture);
        JsonElement root = doc.RootElement;
        int steps = root.GetProperty("steps").GetInt32();
        double shift = root.GetProperty("shift").GetDouble();
        (float[] sigmas, _) = SanaWmSchedule.Build(steps, shift);

        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler();
        scheduler.SetSigmas(sigmas.AsSpan(0, steps));

        Assert.Equal(steps, scheduler.NumInferenceSteps);
        Assert.True(scheduler.HasExplicitSigmas);
        AssertSameBits(ReadFloats(root, "sigmas_float32"), scheduler.Sigmas(), "scheduler sigmas");
        AssertSameBits(ReadFloats(root, "timesteps_float32"), scheduler.Timesteps.ToArray(), "scheduler timesteps");
        Assert.Equal(0.0f, scheduler.SigmaAt(steps));
    }

    [Theory]
    [InlineData(0, 9.8)]
    [InlineData(-3, 9.8)]
    [InlineData(60, 0.0)]
    [InlineData(60, double.NaN)]
    public void Build_RejectsInvalidArguments(int steps, double shift)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => SanaWmSchedule.Build(steps, shift));
    }

    [Fact]
    public void EulerReplay_ConstantVelocity_MovesSampleByDeltaSigmaTimesVelocity()
    {
        (float[] sigmas, _) = SanaWmSchedule.Build(2, 9.8);
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler();
        scheduler.SetSigmas(sigmas.AsSpan(0, 2));

        // Synthetic model: the same velocity at every step. diffusers' step is x_next = x + (sigma_next - sigma) * v.
        float[] start = { 0.5f, -1.0f, 2.0f, 0.0f };
        float[] velocity = { 1.0f, -2.0f, 0.5f, 3.0f };
        using Tensor v = new Tensor(new TensorShape([start.Length]), DType.F32);
        using Tensor x0 = new Tensor(new TensorShape([start.Length]), DType.F32);
        using Tensor x1 = new Tensor(new TensorShape([start.Length]), DType.F32);
        using Tensor x2 = new Tensor(new TensorShape([start.Length]), DType.F32);
        velocity.CopyTo(v.AsSpan<float>());
        start.CopyTo(x0.AsSpan<float>());

        scheduler.Step(x1, v, x0, 0);
        scheduler.Step(x2, v, x1, 1);

        Assert.True(scheduler.Dt(0) < 0f, "sigma decreases, so the flow-match step size is negative");
        for (int i = 0; i < start.Length; i++)
        {
            double expected = start[i] + velocity[i] * (sigmas[1] - sigmas[0]) + velocity[i] * (0.0 - sigmas[1]);
            Assert.True(Math.Abs(x2.AsSpan<float>()[i] - expected) <= 1e-6, $"element {i}: expected {expected}, got {x2.AsSpan<float>()[i]}");
            Assert.True(Math.Abs(x2.AsSpan<float>()[i] - (start[i] - velocity[i])) <= 1e-6, "sigma0 is 1, so the replay lands at x0 - v");
        }
    }

    private static JsonDocument LoadFixture(string name) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", name)));

    private static float[] ReadFloats(JsonElement root, string property)
    {
        JsonElement array = root.GetProperty(property);
        float[] values = new float[array.GetArrayLength()];
        int i = 0;
        foreach (JsonElement element in array.EnumerateArray())
        {
            values[i++] = (float)element.GetDouble();
        }
        return values;
    }

    private static void AssertSameBits(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{name}[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
        }
    }
}
