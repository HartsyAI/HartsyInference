using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Integration lane: the Paint schedule, both samplers and the 2.1 guidance against diffusers itself (<c>UniPCMultistepScheduler</c>,
/// <c>EulerAncestralDiscreteScheduler</c>) and upstream's guidance code. The fixture comes from <c>tests/python-reference/dump_paint_sampler.py</c>;
/// set <c>HARTSY_PAINT_SAMPLER_REF</c> to it. The fixture needs no model weights. A fixed synthetic model drives both sides, so this compares the scheduler arithmetic.</summary>
[Trait("Category", "Integration")]
public sealed class HunyuanPaintSamplerReferenceTests(ITestOutputHelper output)
{
    private static unsafe float[] Read(Dictionary<string, Tensor> fixture, string key)
    {
        Tensor t = fixture[key];
        float[] values = new float[t.ElementCount];
        new ReadOnlySpan<float>((float*)t.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    // The model of the fixture, in float32 like the oracle: v(x, t) = 0.5 tanh(x) - 0.2 x cos(t / 150) + 0.1 sin(t / 37).
    private static float[] Model(float[] x, int t)
    {
        float cos = (float)Math.Cos(t / 150.0), sin = (float)Math.Sin(t / 37.0);
        float[] v = new float[x.Length];
        for (int i = 0; i < v.Length; i++)
            v[i] = 0.5f * MathF.Tanh(x[i]) - 0.2f * x[i] * cos + 0.1f * sin;
        return v;
    }

    private static (double MaxAbs, double RelRms) Compare(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double max = 0, diff2 = 0, ref2 = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - (double)expected[i];
            max = Math.Max(max, Math.Abs(d));
            diff2 += d * d;
            ref2 += (double)expected[i] * expected[i];
        }
        return (max, Math.Sqrt(diff2 / Math.Max(ref2, 1e-30)));
    }

    private static Dictionary<string, Tensor> Load(SafeTensorsLoader loader)
    {
        loader.Load(Environment.GetEnvironmentVariable("HARTSY_PAINT_SAMPLER_REF")!);
        return loader.GetAllTensors();
    }

    [RealWeightsFact("HARTSY_PAINT_SAMPLER_REF")]
    public void Schedule_MatchesDiffusers()
    {
        using SafeTensorsLoader loader = new();
        Dictionary<string, Tensor> fixture = Load(loader);
        HunyuanPaintSchedule schedule = new();
        float[] alphas = Read(fixture, "alphas_cumprod");
        // diffusers builds the table in float32, and near the end (alpha ~ 1e-7, after the zero-SNR rescale cancels almost everything) its own rounding is the
        // difference: the tolerance is 1e-4 relative, and the table here is float64.
        double worstRelative = 0;
        int worstIndex = 0;
        for (int i = 0; i < alphas.Length; i++)
        {
            double relative = Math.Abs(schedule.AlphasCumprod[i] - alphas[i]) / alphas[i];
            if (relative > worstRelative)
                (worstRelative, worstIndex) = (relative, i);
        }
        output.WriteLine($"alphas_cumprod: 1000 entries, worst relative difference {worstRelative:E3} at timestep {worstIndex} (alpha {alphas[worstIndex]:E3})");
        Assert.True(worstRelative < 1e-4, $"alphas_cumprod differ by {worstRelative:E3}");

        foreach (int steps in new[] { 15, 30 })
        {
            foreach (string kind in new[] { "unipc", "euler" })
            {
                float[] timesteps = Read(fixture, $"{kind}.{steps}.timesteps"), sigmas = Read(fixture, $"{kind}.{steps}.sigmas");
                Assert.Equal(timesteps.Select(t => (int)t), HunyuanPaintSchedule.TrailingTimesteps(steps));
                double[] mine = schedule.SigmasAt(HunyuanPaintSchedule.TrailingTimesteps(steps), 0.0);
                Assert.Equal(sigmas.Length, mine.Length);
                double worst = 0;
                for (int i = 0; i < mine.Length; i++)
                    worst = Math.Max(worst, Math.Abs(mine[i] - sigmas[i]) / Math.Max(sigmas[i], 1.0));
                output.WriteLine($"{kind} {steps} steps: timesteps equal, sigmas worst relative difference {worst:E3}");
                Assert.True(worst < 1e-5, $"{kind} {steps}-step sigmas differ by {worst:E3}");
            }
        }
    }

    [RealWeightsFact("HARTSY_PAINT_SAMPLER_REF")]
    public void UniPc_MatchesDiffusers()
    {
        using SafeTensorsLoader loader = new();
        Dictionary<string, Tensor> fixture = Load(loader);
        float[] start = Read(fixture, "init_sample");
        foreach (int steps in new[] { 15, 30 })
        {
            float[] rows = Read(fixture, $"unipc.{steps}.sample");
            HunyuanPaintUniPcSampler sampler = new(new HunyuanPaintSchedule(), steps);
            float[] sample = start;
            double worst = 0;
            for (int step = 0; step < steps; step++)
            {
                sample = sampler.Step(Model(sampler.ScaleModelInput(sample, step), sampler.Timesteps[step]), sample, step, null);
                (_, double rel) = Compare(sample, rows.AsSpan((step + 1) * start.Length, start.Length));
                worst = Math.Max(worst, rel);
                if (step == steps - 1 || step < 3)
                    output.WriteLine($"unipc {steps} steps, after step {step + 1}: relRMS {rel:E3}");
            }
            output.WriteLine($"unipc {steps} steps: worst relRMS over the trajectory {worst:E3}");
            Assert.True(worst < 1e-4, $"UniPC {steps} steps drifts from diffusers by {worst:E3}");
        }
    }

    [RealWeightsFact("HARTSY_PAINT_SAMPLER_REF")]
    public void EulerAncestral_MatchesDiffusers()
    {
        using SafeTensorsLoader loader = new();
        Dictionary<string, Tensor> fixture = Load(loader);
        float[] start = Read(fixture, "init_sample");
        double initSigma = Read(fixture, "euler.init_noise_sigma")[0];
        foreach (int steps in new[] { 15, 30 })
        {
            float[] rows = Read(fixture, $"euler.{steps}.sample"), inputs = Read(fixture, $"euler.{steps}.model_input"), noises = Read(fixture, $"euler.{steps}.noise");
            HunyuanPaintEulerAncestralSampler sampler = new(new HunyuanPaintSchedule(), steps);
            Assert.Equal(initSigma, sampler.InitialNoiseScale, 3);
            float[] sample = start;
            double worstInput = 0, worst = 0;
            for (int step = 0; step < steps; step++)
            {
                float[] scaled = sampler.ScaleModelInput(sample, step);
                worstInput = Math.Max(worstInput, Compare(scaled, inputs.AsSpan(step * start.Length, start.Length)).RelRms);
                sample = sampler.Step(Model(scaled, sampler.Timesteps[step]), sample, step, noises.AsSpan(step * start.Length, start.Length).ToArray());
                worst = Math.Max(worst, Compare(sample, rows.AsSpan((step + 1) * start.Length, start.Length)).RelRms);
            }
            output.WriteLine($"euler ancestral {steps} steps: init sigma {sampler.InitialNoiseScale:F3} (diffusers {initSigma:F3}), model input worst relRMS {worstInput:E3}, sample worst relRMS {worst:E3}");
            Assert.True(worstInput < 1e-5 && worst < 1e-4, $"Euler ancestral {steps} steps: input {worstInput:E3}, sample {worst:E3}");
        }
    }

    [RealWeightsFact("HARTSY_PAINT_SAMPLER_REF")]
    public void Guidance_MatchesUpstream()
    {
        using SafeTensorsLoader loader = new();
        Dictionary<string, Tensor> fixture = Load(loader);
        float[] azimuths = Read(fixture, "guidance.azimuths"), expectedScales = Read(fixture, "guidance.view_scale"), branches = Read(fixture, "guidance.branches"), expected = Read(fixture, "guidance.result");
        float guidance = Read(fixture, "guidance.scale")[0];
        const int materials = 2, views = 8, latent = 8;
        int per = views * latent * latent * 4;
        float[] scales = azimuths.Select(a => HunyuanPaintPipeline.ViewScale(a)).ToArray();
        // The oracle's scales repeat per material, so the first `views` entries are all there is to check.
        for (int v = 0; v < views; v++)
            Assert.Equal(expectedScales[v], scales[v], 6);

        // The oracle runs both materials in one batch ([material * view] order, uncond | ref | full chunks); the C# side guides one material at a time from
        // the unconditional and the full branch (u + g s (ref - u) + g s (full - ref) = u + g s (full - u)).
        float[] result = new float[materials * per];
        for (int m = 0; m < materials; m++)
        {
            float[] uncond = branches.AsSpan(m * per, per).ToArray(), full = branches.AsSpan(2 * materials * per + m * per, per).ToArray();
            HunyuanPaintPipeline.Guide([uncond, full], scales, guidance, views, latent).CopyTo(result, m * per);
        }
        (double max, double rel) = Compare(result, expected);
        output.WriteLine($"guidance: {materials} materials x {views} views, view scales equal, max|diff| {max:E3}, relRMS {rel:E3}");
        Assert.True(rel < 1e-6, $"guidance differs from upstream by relRMS {rel:E3}");
    }
}
