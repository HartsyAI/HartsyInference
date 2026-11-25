using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.ModelAssets.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

public sealed class DuoRopeAndSamplerTests
{
    [Fact]
    public void InverseFrequencies_AreTheUpstreamSchedule()
    {
        float[] inv = DuoRope.InverseFrequencies();
        Assert.Equal(16, inv.Length);
        Assert.Equal(1f, inv[0]);
        Assert.Equal(MathF.Pow(10000f, -1f / 16f), inv[1], 1e-6f);
        Assert.Equal(MathF.Pow(10000f, -15f / 16f), inv[15], 1e-6f);
    }

    [Fact]
    public void Tables_RotateTheThreeAxesIntoTheirOwnSlotsAndDuplicateTheHalves()
    {
        double[] positions = [1.0, 0.0, 0.0];
        (float[] cos, float[] sin) = DuoRope.Tables(positions, headDim: 128);
        Assert.Equal(MathF.Cos(1f), cos[0], 1e-6f);
        Assert.Equal(MathF.Sin(1f), sin[0], 1e-6f);
        // Slot 1 of the t axis: angle = 1 * 10000^(-1/16); its partner sits at +48.
        float angle1 = MathF.Pow(10000f, -1f / 16f);
        Assert.Equal(MathF.Cos(angle1), cos[1], 1e-6f);
        Assert.Equal(cos[1], cos[49], 0f);
        Assert.Equal(sin[1], sin[49], 0f);
        // The h and w axes have zero position here, so their slots are cos 1 and sin 0.
        Assert.Equal(1f, cos[16], 0f);
        Assert.Equal(0f, sin[16], 0f);
        // Channels past the 96 rotary channels are never set.
        for (int c = 96; c < 128; c++)
        {
            Assert.Equal(0f, cos[c]);
            Assert.Equal(0f, sin[c]);
        }
    }

    [Fact]
    public void Tables_CastPositionsToF32BeforeTheAngle()
    {
        // A clock value that is not representable in F32 must be cast first, as upstream does.
        double position = 1.0 + 1e-9;
        (float[] cos, _) = DuoRope.Tables([position, 0.0, 0.0], headDim: 128);
        Assert.Equal(MathF.Cos((float)position), cos[0], 1e-6f);
    }

    [Fact]
    public void Tables_RefuseAHeadDimTooSmallForTheRotaryChannels()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => DuoRope.Tables([0.0, 0.0, 0.0], headDim: 64));
    }

    [Fact]
    public void Euler_IntegratesAConstantVelocityExactly()
    {
        float[] x = [0f, 0f];
        float[] velocity = [2f, -1f];
        float[] schedule = DuoSchedule.Grid(4, 1.0);
        for (int i = 0; i < 4; i++)
        {
            DuoSampler.Euler(x, velocity, schedule[i], schedule[i + 1], x);
        }
        Assert.Equal(2f, x[0], 1e-6f);
        Assert.Equal(-1f, x[1], 1e-6f);
    }

    [Fact]
    public void ToX0_OfAPerfectVelocityIsTheCleanSample()
    {
        float[] clean = [1f, -2f, 0.5f];
        float[] noise = [0.25f, 0.75f, -1f];
        float t = 0.375f;
        float[] noisy = new float[3];
        DuoSampler.AddNoise(clean, noise, t, noisy);
        float[] velocity = new float[3];
        for (int i = 0; i < 3; i++)
        {
            velocity[i] = clean[i] - noise[i];
        }
        float[] x0 = new float[3];
        DuoSampler.ToX0(noisy, velocity, t, x0);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(clean[i], x0[i], 1e-6f);
        }
    }

    [Fact]
    public void AddNoise_HitsTheNoiseAtZeroAndTheCleanAtOne()
    {
        float[] clean = [3f];
        float[] noise = [-4f];
        float[] dst = new float[1];
        DuoSampler.AddNoise(clean, noise, 0f, dst);
        Assert.Equal(-4f, dst[0]);
        DuoSampler.AddNoise(clean, noise, 1f, dst);
        Assert.Equal(3f, dst[0]);
    }

    [Fact]
    public void Sampler_RefusesSpansOfDifferentLengths()
    {
        Assert.Throws<ArgumentException>(() => DuoSampler.Euler(new float[2], new float[3], 0f, 1f, new float[2]));
    }

    [Fact]
    public void ConsistencyLoop_EndsOnTheLastX0AfterTheFinalStep()
    {
        // Distill schedule: four steps. With a perfect velocity the x0 estimate of each step equals the clean sample.
        float[] clean = [0.5f, -0.5f];
        float[] noise = [1f, 1f];
        float[] schedule = DuoSchedule.Grid(DuoSchedule.DistillSteps, DuoSchedule.DistillShift);
        float[] state = new float[2];
        DuoSampler.AddNoise(clean, noise, schedule[0], state);
        float[] velocity = [clean[0] - noise[0], clean[1] - noise[1]];
        float[] x0 = new float[2];
        for (int i = 0; i < DuoSchedule.DistillSteps; i++)
        {
            DuoSampler.ToX0(state, velocity, schedule[i], x0);
            Assert.Equal(clean[0], x0[0], 1e-5f);
            Assert.Equal(clean[1], x0[1], 1e-5f);
            if (i + 1 < DuoSchedule.DistillSteps)
            {
                DuoSampler.AddNoise(clean, noise, schedule[i + 1], state);
            }
        }
    }
}
