using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Host-side SANA-WM camera geometry (ray frames and their SE(3) inverses) and the camera RoPE column slice.</summary>
public sealed class SanaWmCamGeometryTests
{
    private const float Tolerance = 1e-6f;

    [Fact]
    public void IdentityCamera_HasZeroTranslation_AndIdentityAtPrincipalRay()
    {
        const int height = 3;
        const int width = 3;
        float[] conditions = Conditions(frames: 1, fx: 2f, fy: 2f, cx: 1f, cy: 1f);
        float[] p = new float[height * width * 16];
        float[] pt = new float[p.Length];
        float[] pi = new float[p.Length];

        SanaWmCamRays.Build(conditions, 1, 1, height, width, p, pt, pi);

        for (int token = 0; token < height * width; token++)
        {
            Assert.Equal(0f, p[token * 16 + 3]);
            Assert.Equal(0f, p[token * 16 + 7]);
            Assert.Equal(0f, p[token * 16 + 11]);
        }
        int principal = 1 * width + 1;
        for (int i = 0; i < 16; i++)
        {
            float expected = i % 5 == 0 ? 1f : 0f;
            Assert.True(MathF.Abs(p[principal * 16 + i] - expected) <= Tolerance, $"principal ray entry {i}");
        }
    }

    [Fact]
    public void RayDirections_AreUnitLength()
    {
        const int batch = 2;
        const int frames = 2;
        const int height = 3;
        const int width = 4;
        float[] conditions = RandomConditions(batch, frames, seed: 7);
        int blocks = batch * frames * height * width;
        float[] p = new float[blocks * 16];
        float[] pt = new float[p.Length];
        float[] pi = new float[p.Length];

        SanaWmCamRays.Build(conditions, batch, frames, height, width, p, pt, pi);

        for (int token = 0; token < blocks; token++)
        {
            float zx = p[token * 16 + 8], zy = p[token * 16 + 9], zz = p[token * 16 + 10];
            Assert.True(MathF.Abs(MathF.Sqrt(zx * zx + zy * zy + zz * zz) - 1f) <= Tolerance, $"token {token}");
        }
    }

    [Fact]
    public void InverseTimesForward_IsIdentity_ForRandomPoses()
    {
        const int batch = 2;
        const int frames = 3;
        const int height = 2;
        const int width = 3;
        float[] conditions = RandomConditions(batch, frames, seed: 11);
        int blocks = batch * frames * height * width;
        float[] p = new float[blocks * 16];
        float[] pt = new float[p.Length];
        float[] pi = new float[p.Length];

        SanaWmCamRays.Build(conditions, batch, frames, height, width, p, pt, pi);

        for (int token = 0; token < blocks; token++)
        {
            Assert.True(MaxDeviationFromIdentity(p.AsSpan(token * 16, 16), pi.AsSpan(token * 16, 16)) <= Tolerance, $"P^-1 P token {token}");
            Assert.True(MaxDeviationFromIdentity(pi.AsSpan(token * 16, 16), p.AsSpan(token * 16, 16)) <= Tolerance, $"P P^-1 token {token}");
        }
    }

    [Fact]
    public void CameraRope_SlicesTheTemporalHeightWidthColumns()
    {
        const int tokens = 2;
        const int headDim = 112;
        float[] cos = new float[tokens * 56];
        float[] sin = new float[tokens * 56];
        for (int token = 0; token < tokens; token++)
        {
            for (int column = 0; column < 56; column++)
            {
                cos[token * 56 + column] = column;
                sin[token * 56 + column] = 100 + column;
            }
        }
        using CpuBackend backend = new();
        using Tensor cosTable = Host(cos, tokens, 56);
        using Tensor sinTable = Host(sin, tokens, 56);

        (Tensor Cos, Tensor Sin) tables = SanaWmCamRope.BuildCameraTables(backend, cosTable, sinTable, headDim);
        using Tensor camCos = tables.Cos;
        using Tensor camSin = tables.Sin;

        int[] source = new int[28];
        int next = 0;
        for (int c = 0; c < 10; c++) source[next++] = c;
        for (int c = 20; c < 29; c++) source[next++] = c;
        for (int c = 38; c < 47; c++) source[next++] = c;
        ReadOnlySpan<float> outCos = camCos.AsReadOnlySpan<float>();
        ReadOnlySpan<float> outSin = camSin.AsReadOnlySpan<float>();
        Assert.Equal(tokens * 56, outCos.Length);
        for (int token = 0; token < tokens; token++)
        {
            for (int pair = 0; pair < 28; pair++)
            {
                Assert.Equal((float)source[pair], outCos[token * 56 + 2 * pair]);
                Assert.Equal((float)source[pair], outCos[token * 56 + 2 * pair + 1]);
                Assert.Equal((float)(100 + source[pair]), outSin[token * 56 + 2 * pair]);
                Assert.Equal((float)(100 + source[pair]), outSin[token * 56 + 2 * pair + 1]);
            }
        }
    }

    private static Tensor Host(float[] values, int rows, int columns)
    {
        Tensor tensor = new(new TensorShape(rows, columns), DType.F32, DeviceKind.Cpu);
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static float MaxDeviationFromIdentity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float max = 0f;
        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                float sum = 0f;
                for (int k = 0; k < 4; k++)
                    sum += a[row * 4 + k] * b[k * 4 + col];
                max = MathF.Max(max, MathF.Abs(sum - (row == col ? 1f : 0f)));
            }
        }
        return max;
    }

    private static float[] Conditions(int frames, float fx, float fy, float cx, float cy)
    {
        float[] values = new float[frames * SanaWmCamRays.ConditionWidth];
        for (int t = 0; t < frames; t++)
        {
            int at = t * SanaWmCamRays.ConditionWidth;
            Identity().CopyTo(values, at);
            values[at + 16] = fx; values[at + 17] = fy; values[at + 18] = cx; values[at + 19] = cy;
        }
        return values;
    }

    private static float[] RandomConditions(int batch, int frames, int seed)
    {
        Random random = new(seed);
        float[] values = new float[batch * frames * SanaWmCamRays.ConditionWidth];
        for (int i = 0; i < batch * frames; i++)
        {
            int at = i * SanaWmCamRays.ConditionWidth;
            RandomPose(random).CopyTo(values, at);
            values[at + 16] = 2.5f + 0.5f * (float)random.NextDouble();
            values[at + 17] = 2.5f + 0.5f * (float)random.NextDouble();
            values[at + 18] = 1.0f + 0.5f * (float)random.NextDouble();
            values[at + 19] = 0.5f + 0.5f * (float)random.NextDouble();
        }
        return values;
    }

    private static float[] Identity()
    {
        float[] pose = new float[16];
        pose[0] = 1f; pose[5] = 1f; pose[10] = 1f; pose[15] = 1f;
        return pose;
    }

    /// <summary>A camera-to-world pose with a uniformly random rotation (from a unit quaternion) and translation.</summary>
    private static float[] RandomPose(Random random)
    {
        double w = random.NextDouble() - 0.5, x = random.NextDouble() - 0.5, y = random.NextDouble() - 0.5, z = random.NextDouble() - 0.5;
        double norm = Math.Sqrt(w * w + x * x + y * y + z * z);
        w /= norm; x /= norm; y /= norm; z /= norm;
        float[] pose = new float[16];
        pose[0] = (float)(1 - 2 * (y * y + z * z)); pose[1] = (float)(2 * (x * y - z * w)); pose[2] = (float)(2 * (x * z + y * w));
        pose[4] = (float)(2 * (x * y + z * w)); pose[5] = (float)(1 - 2 * (x * x + z * z)); pose[6] = (float)(2 * (y * z - x * w));
        pose[8] = (float)(2 * (x * z - y * w)); pose[9] = (float)(2 * (y * z + x * w)); pose[10] = (float)(1 - 2 * (x * x + y * y));
        pose[3] = (float)(random.NextDouble() * 2 - 1);
        pose[7] = (float)(random.NextDouble() * 2 - 1);
        pose[11] = (float)(random.NextDouble() * 2 - 1);
        pose[15] = 1f;
        return pose;
    }
}
