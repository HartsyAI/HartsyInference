using HartsyInference.World.Camera;
using Xunit;
using static HartsyInference.World.Tests.Camera.CameraTestData;

namespace HartsyInference.World.Tests.Camera;

public sealed class Se3MathTests
{
    [Fact]
    public void Invert_RandomRigidPose_ComposesToIdentity()
    {
        Random rng = new(3);
        for (int trial = 0; trial < 50; trial++)
        {
            float[] pose = RandomPose(rng);
            float[] inverse = new float[16];
            Se3Math.Invert(pose, inverse);

            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                {
                    float v = 0f;
                    for (int k = 0; k < 4; k++) v += pose[r * 4 + k] * inverse[k * 4 + c];
                    AssertClose(r == c ? 1f : 0f, v, 1e-5f);
                }
        }
    }
}
