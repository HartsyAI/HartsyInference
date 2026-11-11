namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// The closed-form <c>ScaleShiftInvariant</c> pose decode of the released config. The pose latents become a rotation (6D with a fixed
/// mean and std, Gram-Schmidt, columns <c>[b1 b2 b3]</c>), a scale (<c>exp</c>) and a translation; the result is post-composed with
/// the pointmap's scale-and-shift transform and reported as a unit quaternion, a translation and one uniform scale. PyTorch3D's
/// row-vector convention is kept: the rotation matrix acts on row vectors, and the matrix of the whole transform is
/// <c>diag(s) · R · diag(scene_scale)</c> with the translation <c>t · diag(scene_scale) + scene_shift</c>.
/// </summary>
public static class SamPoseDecoder
{
    /// <summary><c>ROTATION_6D_MEAN</c> of <c>inference_utils.py</c>.</summary>
    public static readonly float[] Rotation6dMean =
        [-0.06366084883674913f, 0.008438224692279752f, 0.00017084786438302483f, 0.0007126610473540038f, -0.0030916726538816417f, 0.5166093753457688f];

    /// <summary><c>ROTATION_6D_STD</c> of <c>inference_utils.py</c>.</summary>
    public static readonly float[] Rotation6dStd =
        [0.6656971967514863f, 0.6787012271867754f, 0.30345010594844524f, 0.4394504420678794f, 0.39817973931717104f, 0.6176286868761914f];

    /// <summary>
    /// Decodes one object. <paramref name="rotation6dNormalized"/> has 6 values, <paramref name="translation"/> and
    /// <paramref name="logScale"/> 3 each; <paramref name="sceneScale"/> and <paramref name="sceneShift"/> are the pointmap scale and shift.
    /// <paramref name="downsampleFactor"/> is the factor <see cref="SamSparseStructure.Downsample"/> returned (1 when it did nothing).
    /// </summary>
    public static SamPose Decode(
        ReadOnlySpan<float> rotation6dNormalized, ReadOnlySpan<float> translation, ReadOnlySpan<float> logScale,
        ReadOnlySpan<float> sceneScale, ReadOnlySpan<float> sceneShift, float downsampleFactor = 1f)
    {
        if (rotation6dNormalized.Length != 6 || translation.Length != 3 || logScale.Length != 3 || sceneScale.Length != 3 || sceneShift.Length != 3)
            throw new ArgumentException("Expected 6 rotation values and 3 each of translation, log scale, scene scale and scene shift.");
        Span<double> r = stackalloc double[9];
        RotationFromSixD(rotation6dNormalized, r);

        // PyTorch3D: scale, then rotate (row vectors), then translate, then the scene scale and shift. M = S · R · diag(sceneScale).
        Span<double> a = stackalloc double[9];
        Span<double> scale = stackalloc double[3];
        Span<double> rotation = stackalloc double[9];
        for (int i = 0; i < 3; i++)
        {
            double s = Math.Exp(logScale[i]);
            double norm = 0.0;
            for (int j = 0; j < 3; j++)
            {
                a[i * 3 + j] = s * r[i * 3 + j] * sceneScale[j];
                norm += a[i * 3 + j] * a[i * 3 + j];
            }

            scale[i] = Math.Sqrt(norm);
            for (int j = 0; j < 3; j++)
                rotation[i * 3 + j] = scale[i] > 0.0 ? a[i * 3 + j] / scale[i] : r[i * 3 + j];
        }

        double meanScale = (scale[0] + scale[1] + scale[2]) / 3.0 * downsampleFactor;
        (double w, double x, double y, double z) = MatrixToQuaternion(rotation);
        return new SamPose(
            (float)w, (float)x, (float)y, (float)z,
            translation[0] * sceneScale[0] + sceneShift[0], translation[1] * sceneScale[1] + sceneShift[1], translation[2] * sceneScale[2] + sceneShift[2],
            (float)meanScale, (float)meanScale, (float)meanScale);
    }

    /// <summary>Row-major 3x3 matrix whose columns are the Gram-Schmidt basis of the de-normalised 6D rotation.</summary>
    internal static void RotationFromSixD(ReadOnlySpan<float> normalized, Span<double> matrix)
    {
        Span<double> v = stackalloc double[6];
        for (int i = 0; i < 6; i++)
            v[i] = normalized[i] * (double)Rotation6dStd[i] + Rotation6dMean[i];
        double[] b1 = Normalize(v[0], v[1], v[2]);
        double dot = b1[0] * v[3] + b1[1] * v[4] + b1[2] * v[5];
        double[] b2 = Normalize(v[3] - dot * b1[0], v[4] - dot * b1[1], v[5] - dot * b1[2]);
        double[] b3 = [b1[1] * b2[2] - b1[2] * b2[1], b1[2] * b2[0] - b1[0] * b2[2], b1[0] * b2[1] - b1[1] * b2[0]];
        for (int row = 0; row < 3; row++)
        {
            matrix[row * 3 + 0] = b1[row];
            matrix[row * 3 + 1] = b2[row];
            matrix[row * 3 + 2] = b3[row];
        }
    }

    /// <summary>PyTorch3D's <c>matrix_to_quaternion</c> (w first) with the real part made non-negative.</summary>
    internal static (double W, double X, double Y, double Z) MatrixToQuaternion(ReadOnlySpan<double> m)
    {
        double m00 = m[0], m01 = m[1], m02 = m[2], m10 = m[3], m11 = m[4], m12 = m[5], m20 = m[6], m21 = m[7], m22 = m[8];
        double w2 = 1.0 + m00 + m11 + m22, x2 = 1.0 + m00 - m11 - m22, y2 = 1.0 - m00 + m11 - m22, z2 = 1.0 - m00 - m11 + m22;
        double w = Math.Sqrt(Math.Max(0.0, w2)) / 2.0, x = Math.Sqrt(Math.Max(0.0, x2)) / 2.0;
        double y = Math.Sqrt(Math.Max(0.0, y2)) / 2.0, z = Math.Sqrt(Math.Max(0.0, z2)) / 2.0;
        double best = Math.Max(Math.Max(w, x), Math.Max(y, z));
        double qw, qx, qy, qz;
        if (best == w)
        {
            qw = w;
            qx = (m21 - m12) / (4.0 * w);
            qy = (m02 - m20) / (4.0 * w);
            qz = (m10 - m01) / (4.0 * w);
        }
        else if (best == x)
        {
            qx = x;
            qw = (m21 - m12) / (4.0 * x);
            qy = (m10 + m01) / (4.0 * x);
            qz = (m02 + m20) / (4.0 * x);
        }
        else if (best == y)
        {
            qy = y;
            qw = (m02 - m20) / (4.0 * y);
            qx = (m10 + m01) / (4.0 * y);
            qz = (m21 + m12) / (4.0 * y);
        }
        else
        {
            qz = z;
            qw = (m10 - m01) / (4.0 * z);
            qx = (m02 + m20) / (4.0 * z);
            qy = (m21 + m12) / (4.0 * z);
        }

        double norm = Math.Sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        double sign = qw < 0.0 ? -1.0 : 1.0;
        return (sign * qw / norm, sign * qx / norm, sign * qy / norm, sign * qz / norm);
    }

    private static double[] Normalize(double x, double y, double z)
    {
        double norm = Math.Max(Math.Sqrt(x * x + y * y + z * z), 1e-12);
        return [x / norm, y / norm, z / norm];
    }
}
