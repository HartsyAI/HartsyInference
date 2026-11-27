namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Per-token UCPE ray transforms from camera-to-world poses in the upstream ray frame: <c>z = normalize(R·d)</c>,
/// <c>x = normalize(R[:,1] × z)</c>, <c>y = normalize(z × x)</c>, and <c>P = [[R_l2w^T, −R_l2w^T·t], [0, 1]]</c>.</summary>
public static class SanaWmCamRays
{
    /// <summary>Floats per frame of camera conditions: a row-major camera-to-world pose, then fx, fy, cx, cy in latent pixels.</summary>
    public const int ConditionWidth = 20;

    private const float NormalizeEpsilon = 1e-6f;

    /// <summary>Writes P, P^T and P^-1 for every token as row-major 4×4 blocks. Tokens run frame, then row, then column,
    /// and pixel centers sit on integer coordinates.</summary>
    public static void Build(
        ReadOnlySpan<float> cameraConditions,
        int batch,
        int frames,
        int height,
        int width,
        Span<float> matrices,
        Span<float> transposed,
        Span<float> inverse)
    {
        if (batch <= 0 || frames <= 0 || height <= 0 || width <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "Batch and grid extents must be positive.");
        if (cameraConditions.Length != batch * frames * ConditionWidth)
            throw new ArgumentException(
                $"Camera conditions must hold {batch * frames * ConditionWidth} floats; got {cameraConditions.Length}.",
                nameof(cameraConditions));
        int tokens = frames * height * width;
        int blocks = batch * tokens * 16;
        if (matrices.Length < blocks || transposed.Length < blocks || inverse.Length < blocks)
            throw new ArgumentException($"Each output must hold {blocks} floats.");

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < frames; t++)
            {
                ReadOnlySpan<float> pose = cameraConditions.Slice((b * frames + t) * ConditionWidth, ConditionWidth);
                for (int row = 0; row < height; row++)
                {
                    for (int col = 0; col < width; col++)
                    {
                        int offset = (b * tokens + (t * height + row) * width + col) * 16;
                        Span<float> p = matrices.Slice(offset, 16);
                        WriteRay(pose, col, row, p);
                        Transpose(p, transposed.Slice(offset, 16));
                        Inverse(p, inverse.Slice(offset, 16));
                    }
                }
            }
        }
    }

    private static void WriteRay(ReadOnlySpan<float> pose, float u, float v, Span<float> p)
    {
        float x = (u - pose[18]) / pose[16];
        float y = (v - pose[19]) / pose[17];
        float scale = 1f / MathF.Sqrt(x * x + y * y + 1f);
        float dx = x * scale, dy = y * scale, dz = scale;

        float wx = pose[0] * dx + pose[1] * dy + pose[2] * dz;
        float wy = pose[4] * dx + pose[5] * dy + pose[6] * dz;
        float wz = pose[8] * dx + pose[9] * dy + pose[10] * dz;
        if (float.IsNaN(wx) || float.IsNaN(wy) || float.IsNaN(wz))
        {
            Identity(p);
            return;
        }

        (float zx, float zy, float zz) = Normalize(wx, wy, wz);
        (float cx, float cy, float cz) = Cross(pose[1], pose[5], pose[9], zx, zy, zz);
        (float xx, float xy, float xz) = Normalize(cx, cy, cz);
        (float ax, float ay, float az) = Cross(zx, zy, zz, xx, xy, xz);
        (float yx, float yy, float yz) = Normalize(ax, ay, az);

        float tx = pose[3], ty = pose[7], tz = pose[11];
        p[0] = xx; p[1] = xy; p[2] = xz; p[3] = -(xx * tx + xy * ty + xz * tz);
        p[4] = yx; p[5] = yy; p[6] = yz; p[7] = -(yx * tx + yy * ty + yz * tz);
        p[8] = zx; p[9] = zy; p[10] = zz; p[11] = -(zx * tx + zy * ty + zz * tz);
        p[12] = 0f; p[13] = 0f; p[14] = 0f; p[15] = 1f;
    }

    private static (float X, float Y, float Z) Normalize(float x, float y, float z)
    {
        float norm = MathF.Max(MathF.Sqrt(x * x + y * y + z * z), NormalizeEpsilon);
        return (x / norm, y / norm, z / norm);
    }

    private static (float X, float Y, float Z) Cross(float ax, float ay, float az, float bx, float by, float bz) =>
        (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);

    private static void Identity(Span<float> p)
    {
        p.Clear();
        p[0] = 1f; p[5] = 1f; p[10] = 1f; p[15] = 1f;
    }

    private static void Transpose(ReadOnlySpan<float> p, Span<float> pt)
    {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                pt[i * 4 + j] = p[j * 4 + i];
    }

    /// <summary>Closed-form SE(3) inverse, as upstream <c>_invert_SE3</c>: <c>R^T</c> and <c>−R^T·t</c>.</summary>
    private static void Inverse(ReadOnlySpan<float> p, Span<float> inv)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
                inv[i * 4 + j] = p[j * 4 + i];
            inv[i * 4 + 3] = -(inv[i * 4] * p[3] + inv[i * 4 + 1] * p[7] + inv[i * 4 + 2] * p[11]);
        }
        inv[12] = 0f; inv[13] = 0f; inv[14] = 0f; inv[15] = 1f;
    }
}
