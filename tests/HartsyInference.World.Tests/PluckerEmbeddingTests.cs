using Xunit;
using HartsyInference.World.Camera;

namespace HartsyInference.World.Tests;

/// <summary>Plücker raymap invariants: unit directions, origin channels = camera position, center ray ≈ optical axis.</summary>
public class PluckerEmbeddingTests
{
    [Fact]
    public void Compute_ProducesUnitDirectionsAndCameraOrigins()
    {
        int h = 8, w = 8;
        float[] pose = new float[16];
        Se3Math.Identity(pose);
        pose[3] = 1.5f; pose[7] = -2.5f; pose[11] = 3.0f;

        float[] rays = new float[h * w * PluckerEmbedding.Channels];
        PluckerEmbedding.Compute(pose, fx: 8, fy: 8, cx: 4, cy: 4, h, w, rays);

        for (int i = 0; i < h * w; i++)
        {
            int b = i * 6;
            Assert.Equal(1.5f, rays[b + 0], 5);
            Assert.Equal(-2.5f, rays[b + 1], 5);
            Assert.Equal(3.0f, rays[b + 2], 5);
            float norm = MathF.Sqrt(rays[b + 3] * rays[b + 3] + rays[b + 4] * rays[b + 4] + rays[b + 5] * rays[b + 5]);
            Assert.True(MathF.Abs(norm - 1f) < 1e-5f, $"direction at pixel {i} not unit: {norm}");
        }

        // Center pixel under an identity pose looks straight down +Z.
        int center = (4 * w + 4) * 6;
        Assert.True(rays[center + 5] > 0.99f, $"center ray z = {rays[center + 5]}");
    }
}
