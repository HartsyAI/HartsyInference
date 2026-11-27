using HartsyInference.World.Camera;
using Xunit;

namespace HartsyInference.World.Tests.Camera;

public sealed class SanaWmActionTests
{
    [Fact]
    public void Parse_ExpandsSegmentsPerFrame_CombiningKeysAndIgnoringWhitespace()
    {
        SanaWmControls[] frames = SanaWmAction.Parse(" w-2 ,DW-1，none-1");

        Assert.Equal(
            [SanaWmControls.Forward, SanaWmControls.Forward, SanaWmControls.Forward | SanaWmControls.YawRight, SanaWmControls.None],
            frames);
    }

    [Theory]
    [InlineData("")]
    [InlineData("w")]
    [InlineData("w-0")]
    [InlineData("w-x")]
    [InlineData("w-")]
    [InlineData("q-3")]
    [InlineData("w-2,,a-1")]
    public void Parse_RejectsMalformedScripts(string script)
    {
        Assert.Throws<FormatException>(() => SanaWmAction.Parse(script));
    }

    [Fact]
    public void Rollout_HasOnePosePerFramePlusTheIdentitySeed()
    {
        float[][] poses = SanaWmAction.ToCameraToWorld("w-3,none-2");

        Assert.Equal(6, poses.Length);
        Assert.Equal([1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f], poses[0]);
    }

    [Fact]
    public void NewPress_SnapsToFullSpeed_AndReleaseCoastsWithTheSlowTimeConstant()
    {
        float[][] poses = SanaWmAction.ToCameraToWorld("w-1,none-1");

        // Upstream divides by (norm + 1e-6), so a full-speed step is slightly short of 0.025.
        double step = SanaWmAction.TranslationPerFrame / (1.0 + 1e-6);
        Assert.Equal((float)step, poses[1][11]);
        double coast = SanaWmAction.TranslationPerFrame * Math.Exp(-1.0 / SanaWmAction.Fps / SanaWmAction.TauCoastSeconds);
        Assert.Equal((float)(step + coast / (1.0 + 1e-6)), poses[2][11]);
    }

    [Fact]
    public void Pitch_SaturatesAtTheLimit()
    {
        float[][] poses = SanaWmAction.ToCameraToWorld("i-200");

        double limit = SanaWmAction.PitchLimitDegrees * Math.PI / 180.0;
        foreach (float[] pose in poses)
            Assert.True(Math.Asin(pose[9]) <= limit + 1e-6);
        Assert.Equal((float)Math.Sin(limit), poses[^1][9], 6);
    }

    [Fact]
    public void Rollout_MatchesUpstreamValuesForTheReferenceScript()
    {
        // Rotation and translation entries [0, 2, 3, 8, 10, 11] of upstream c2w for w-80,dw-40,aw-40 (run B_action161).
        float[][] poses = SanaWmAction.ToCameraToWorld("w-80,dw-40,aw-40");

        Assert.Equal(161, poses.Length);
        AssertEntries(poses[81], [0.9999451637268066f, 0.010471784509718418f, 0.00026179433916695416f, -0.010471784509718418f,
            0.9999451637268066f, 2.024996519088745f]);
        AssertEntries(poses[120], [0.9135454297065735f, 0.4067366421222687f, 0.211477130651474f, -0.4067366421222687f,
            0.9135454297065735f, 2.9699196815490723f]);
        AssertEntries(poses[160], [1.0f, 0f, 0.4127858579158783f, 0f, 1.0f, 3.942002773284912f]);
    }

    [Fact]
    public void Packing_MatchesUpstreamValuesForTheReferenceScript()
    {
        float[][] poses = SanaWmAction.ToCameraToWorld("w-80,dw-40,aw-40");
        PinholeIntrinsics source = new(797.878662109375f, 830.05029296875f, 844.2675170898438f, 463.7225036621094f);
        CoverCropMapping crop = CoverCropMapping.Create(1691, 930, 1280, 704);

        using SanaWmCameraConditioning conditioning = SanaWmCameraConditioning.Prepare(poses, [source], crop, 1280, 704);

        Assert.Equal(21, conditioning.LatentFrames);
        Assert.Equal(21 * SanaWmCameraConditioning.ConditionWidth, conditioning.Conditions.Length);
        ReadOnlySpan<float> row = conditioning.Conditions.AsSpan(20 * SanaWmCameraConditioning.ConditionWidth, 20);
        Assert.Equal(0.4127858579158783f, row[3], 6);
        Assert.Equal(3.942002773284912f, row[11], 6);
        // Upstream maps the crop in single precision; the shared mapping rounds once from double, so allow one ulp.
        float[] latent = [18.873533248901367f, 19.635597229003906f, 19.97084617614746f, 10.969779968261719f];
        for (int i = 0; i < latent.Length; i++)
            Assert.True(MathF.Abs(latent[i] - row[16 + i]) <= 1e-5f * latent[i], $"latent intrinsics {i}: {row[16 + i]}");

        // chunk_plucker[0, c, 20, 5, 7] for channels 0..5, 42 and 47, then [0, c, 0, 0, 0] for channels 0 and 2.
        ReadOnlySpan<float> plucker = conditioning.ChunkPlucker.AsReadOnlySpan<float>();
        int[] channels = [0, 1, 2, 3, 4, 5, 42, 47];
        float[] expected = [-0.4893816411495209f, -0.24304808676242828f, 0.8375160694122314f, 0.915593147277832f,
            -2.1846752166748047f, -0.09899110347032547f, -0.549405574798584f, -0.10032681375741959f];
        for (int i = 0; i < channels.Length; i++)
            Assert.Equal(expected[i], plucker[PluckerIndex(channels[i], 20, 5, 7)], 5);
        Assert.Equal(-0.678550124168396f, plucker[PluckerIndex(0, 0, 0, 0)], 6);
        Assert.Equal(0.6412667036056519f, plucker[PluckerIndex(2, 0, 0, 0)], 6);
    }

    [Fact]
    public void FitOverTime_BroadcastsTruncatesAndInterpolates()
    {
        PinholeIntrinsics a = new(100f, 200f, 50f, 60f);
        PinholeIntrinsics b = new(300f, 400f, 70f, 80f);

        Assert.Equal([a, a, a], SanaWmCameraConditioning.FitOverTime([a], 3));
        Assert.Equal([a], SanaWmCameraConditioning.FitOverTime([a, b], 1));
        PinholeIntrinsics[] fitted = SanaWmCameraConditioning.FitOverTime([a, b], 3);
        Assert.Equal(a, fitted[0]);
        Assert.Equal(new PinholeIntrinsics(200f, 300f, 60f, 70f), fitted[1]);
        Assert.Equal(b, fitted[2]);
    }

    private static int PluckerIndex(int channel, int latentFrame, int y, int x) => ((channel * 21 + latentFrame) * 22 + y) * 40 + x;

    private static void AssertEntries(float[] pose, float[] expected)
    {
        int[] entries = [0, 2, 3, 8, 10, 11];
        for (int i = 0; i < entries.Length; i++)
            Assert.Equal(expected[i], pose[entries[i]], 6);
    }
}
