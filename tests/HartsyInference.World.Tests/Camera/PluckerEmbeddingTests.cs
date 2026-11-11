using HartsyInference.World.Camera;
using Xunit;
using static HartsyInference.World.Tests.Camera.CameraTestData;

namespace HartsyInference.World.Tests.Camera;

public sealed class PluckerEmbeddingTests
{
    private static readonly float[] Identity = [1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f];

    [Fact]
    public void IdentityCamera_CenterPixelZeroOffset_OriginDirection_IsForwardRay()
    {
        float[] ray = CenterRay(PluckerLayout.OriginDirection);
        AssertClose(0f, ray[0], 1e-6f); AssertClose(0f, ray[1], 1e-6f); AssertClose(0f, ray[2], 1e-6f);
        AssertClose(0f, ray[3], 1e-6f); AssertClose(0f, ray[4], 1e-6f); AssertClose(1f, ray[5], 1e-6f);
    }

    [Fact]
    public void IdentityCamera_CenterPixelZeroOffset_MomentDirection_HasZeroMoment()
    {
        float[] ray = CenterRay(PluckerLayout.MomentDirection);
        AssertClose(0f, ray[0], 1e-6f); AssertClose(0f, ray[1], 1e-6f); AssertClose(0f, ray[2], 1e-6f);
        AssertClose(0f, ray[3], 1e-6f); AssertClose(0f, ray[4], 1e-6f); AssertClose(1f, ray[5], 1e-6f);
    }

    [Fact]
    public void IdentityCamera_CenterPixelZeroOffset_DirectionMoment_PutsDirectionFirst()
    {
        float[] ray = CenterRay(PluckerLayout.DirectionMoment);
        AssertClose(0f, ray[0], 1e-6f); AssertClose(0f, ray[1], 1e-6f); AssertClose(1f, ray[2], 1e-6f);
        AssertClose(0f, ray[3], 1e-6f); AssertClose(0f, ray[4], 1e-6f); AssertClose(0f, ray[5], 1e-6f);
    }

    [Fact]
    public void RandomPoses_EveryDirectionIsUnitLength()
    {
        Random rng = new(11);
        for (int trial = 0; trial < 40; trial++)
        {
            int height = rng.Next(1, 17), width = rng.Next(1, 17);
            float[] pose = RandomPose(rng);
            float[] raymap = new float[height * width * PluckerEmbedding.Channels];
            PluckerEmbedding.Compute(pose, 300f, 310f, width * 0.4f, height * 0.6f, height, width, raymap,
                PluckerEmbedding.PixelCenterOffset, PluckerLayout.OriginDirection);
            for (int p = 0; p < height * width; p++)
            {
                int b = p * PluckerEmbedding.Channels;
                float length = MathF.Sqrt(raymap[b + 3] * raymap[b + 3] + raymap[b + 4] * raymap[b + 4] + raymap[b + 5] * raymap[b + 5]);
                AssertClose(1f, length, 1e-5f);
            }
        }
    }

    [Fact]
    public void RandomPoses_MomentEqualsOriginCrossDirection()
    {
        Random rng = new(17);
        for (int trial = 0; trial < 40; trial++)
        {
            int height = rng.Next(1, 13), width = rng.Next(1, 13);
            float[] pose = RandomPose(rng);
            float[] origins = new float[height * width * PluckerEmbedding.Channels];
            float[] moments = new float[origins.Length];
            PluckerEmbedding.Compute(pose, 250f, 260f, width / 2f, height / 2f, height, width, origins,
                PluckerEmbedding.PixelCenterOffset, PluckerLayout.OriginDirection);
            PluckerEmbedding.Compute(pose, 250f, 260f, width / 2f, height / 2f, height, width, moments,
                PluckerEmbedding.PixelCenterOffset, PluckerLayout.MomentDirection);
            for (int p = 0; p < height * width; p++)
            {
                int b = p * PluckerEmbedding.Channels;
                double ox = origins[b], oy = origins[b + 1], oz = origins[b + 2];
                double dx = origins[b + 3], dy = origins[b + 4], dz = origins[b + 5];
                Assert.Equal(pose[3], (float)ox);
                Assert.Equal(pose[7], (float)oy);
                Assert.Equal(pose[11], (float)oz);
                AssertClose((float)(oy * dz - oz * dy), moments[b], 1e-4f);
                AssertClose((float)(oz * dx - ox * dz), moments[b + 1], 1e-4f);
                AssertClose((float)(ox * dy - oy * dx), moments[b + 2], 1e-4f);
                Assert.Equal(origins[b + 3], moments[b + 3]);
                Assert.Equal(origins[b + 4], moments[b + 4]);
                Assert.Equal(origins[b + 5], moments[b + 5]);
            }
        }
    }

    [Fact]
    public void Layouts_AreChannelPermutationsOfTheSameRays()
    {
        Random rng = new(19);
        for (int trial = 0; trial < 20; trial++)
        {
            int height = rng.Next(1, 11), width = rng.Next(1, 11);
            float[] pose = RandomPose(rng);
            float[] origin = Render(pose, height, width, PluckerLayout.OriginDirection);
            float[] moment = Render(pose, height, width, PluckerLayout.MomentDirection);
            float[] directionMoment = Render(pose, height, width, PluckerLayout.DirectionMoment);
            for (int p = 0; p < height * width; p++)
            {
                int b = p * PluckerEmbedding.Channels;
                // Direction is the same ray triple in every layout; DirectionMoment is MomentDirection with the halves swapped.
                Assert.True(SameBits(origin.AsSpan(b + 3, 3), moment.AsSpan(b + 3, 3)));
                Assert.True(SameBits(directionMoment.AsSpan(b, 3), moment.AsSpan(b + 3, 3)));
                Assert.True(SameBits(directionMoment.AsSpan(b + 3, 3), moment.AsSpan(b, 3)));
                Assert.Equal(Sorted(moment.AsSpan(b, 6)), Sorted(directionMoment.AsSpan(b, 6)));
            }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DefaultLayouts_AreByteIdenticalToPreP10Implementation(bool moment)
    {
        Random rng = new(23);
        PluckerLayout layout = moment ? PluckerLayout.MomentDirection : PluckerLayout.OriginDirection;
        for (int trial = 0; trial < 200; trial++)
        {
            int height = rng.Next(0, 17), width = rng.Next(1, 17);
            float[] pose = RandomPose(rng);
            float fx = Uniform(rng, 50f, 1200f), fy = Uniform(rng, 50f, 1200f);
            float cx = Uniform(rng, 0f, width), cy = Uniform(rng, 0f, height);

            float[] expected = new float[height * width * PluckerEmbedding.Channels];
            LegacyCompute(pose, fx, fy, cx, cy, height, width, expected, moment);

            float[] legacyOverload = new float[expected.Length];
            PluckerEmbedding.Compute(pose, fx, fy, cx, cy, height, width, legacyOverload, moment);
            Assert.True(SameBits(expected, legacyOverload), $"bool overload differs at trial {trial}");

            float[] explicitOverload = new float[expected.Length];
            PluckerEmbedding.Compute(pose, fx, fy, cx, cy, height, width, explicitOverload,
                PluckerEmbedding.PixelCenterOffset, layout);
            Assert.True(SameBits(expected, explicitOverload), $"explicit overload differs at trial {trial}");
        }
    }

    [Fact]
    public void PixelOffset_HalfEqualsZeroOffsetWithPrincipalPointShiftedByHalf()
    {
        // All values are dyadic, so both sample positions are exact and the two raymaps must match bit for bit.
        float[] shifted = new float[5 * 5 * PluckerEmbedding.Channels];
        float[] reference = new float[shifted.Length];
        PluckerEmbedding.Compute(Identity, 2f, 2f, 2f, 2f, 5, 5, shifted, 0.5f, PluckerLayout.OriginDirection);
        PluckerEmbedding.Compute(Identity, 2f, 2f, 1.5f, 1.5f, 5, 5, reference, 0f, PluckerLayout.OriginDirection);
        Assert.True(SameBits(reference, shifted));
    }

    [Fact]
    public void UndefinedLayout_IsRejected()
    {
        float[] destination = new float[PluckerEmbedding.Channels];
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PluckerEmbedding.Compute(Identity, 1f, 1f, 0f, 0f, 1, 1, destination, 0.5f, (PluckerLayout)7));
    }

    private static float[] CenterRay(PluckerLayout layout)
    {
        // 4x4 identity camera; with cx = cy = 2 and zero offset, pixel (2, 2) lies on the optical axis.
        float[] raymap = new float[4 * 4 * PluckerEmbedding.Channels];
        PluckerEmbedding.Compute(Identity, 1f, 1f, 2f, 2f, 4, 4, raymap, 0f, layout);
        int center = (2 * 4 + 2) * PluckerEmbedding.Channels;
        return raymap.AsSpan(center, PluckerEmbedding.Channels).ToArray();
    }

    private static float[] Render(float[] pose, int height, int width, PluckerLayout layout)
    {
        float[] raymap = new float[height * width * PluckerEmbedding.Channels];
        PluckerEmbedding.Compute(pose, 180f, 190f, width / 2f, height / 2f, height, width, raymap, 0.5f, layout);
        return raymap;
    }

    private static float[] Sorted(ReadOnlySpan<float> values)
    {
        float[] copy = values.ToArray();
        Array.Sort(copy);
        return copy;
    }

    /// <summary>The pre-P10 <c>PluckerEmbedding.Compute</c>, kept verbatim as the byte-identity oracle for the default layouts.</summary>
    private static void LegacyCompute(ReadOnlySpan<float> cameraToWorld,
        float fx, float fy, float cx, float cy, int height, int width, Span<float> destination, bool moment)
    {
        const int Channels = PluckerEmbedding.Channels;
        if (cameraToWorld.Length < 16) throw new ArgumentException("cameraToWorld must hold 16 floats.", nameof(cameraToWorld));
        if (destination.Length < height * width * Channels)
            throw new ArgumentException($"destination must hold {height * width * Channels} floats.", nameof(destination));
        if (fx == 0 || fy == 0) throw new ArgumentException("focal lengths must be non-zero.");

        float ox = cameraToWorld[3], oy = cameraToWorld[7], oz = cameraToWorld[11];
        for (int yPix = 0; yPix < height; yPix++)
        {
            float ys = (yPix + 0.5f - cy) / fy;
            for (int xPix = 0; xPix < width; xPix++)
            {
                float xs = (xPix + 0.5f - cx) / fx;
                float inv = 1f / MathF.Sqrt(xs * xs + ys * ys + 1f);
                float dxc = xs * inv, dyc = ys * inv, dzc = inv;
                // World direction = R_c2w · d_cam.
                float dwx = cameraToWorld[0] * dxc + cameraToWorld[1] * dyc + cameraToWorld[2] * dzc;
                float dwy = cameraToWorld[4] * dxc + cameraToWorld[5] * dyc + cameraToWorld[6] * dzc;
                float dwz = cameraToWorld[8] * dxc + cameraToWorld[9] * dyc + cameraToWorld[10] * dzc;
                int baseIdx = (yPix * width + xPix) * Channels;
                if (moment)
                {
                    // Plücker moment m = o × d (GameCraft ray_condition layout).
                    destination[baseIdx + 0] = oy * dwz - oz * dwy;
                    destination[baseIdx + 1] = oz * dwx - ox * dwz;
                    destination[baseIdx + 2] = ox * dwy - oy * dwx;
                }
                else
                {
                    destination[baseIdx + 0] = ox;
                    destination[baseIdx + 1] = oy;
                    destination[baseIdx + 2] = oz;
                }
                destination[baseIdx + 3] = dwx;
                destination[baseIdx + 4] = dwy;
                destination[baseIdx + 5] = dwz;
            }
        }
    }
}
