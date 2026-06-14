namespace HartsyInference.Interactive.Camera;

/// <summary>SE(3) helpers shared by the camera-conditioned world models (Matrix-Game 3.0, Hunyuan-GameCraft): rigid
/// 4×4 inverse, Euler→rotation, quaternion slerp, and the action→pose integrator. Matrices are row-major
/// <c>float[16]</c>, camera-to-world unless stated otherwise. Ported from Matrix-Game-3 <c>utils/cam_utils.py</c>;
/// the integrator's step semantics are validation-gated (confirmed against the reference on first run).</summary>
public static class Se3Math
{
    /// <summary>The fixed camera→world axis remap Matrix-Game applies in <c>get_extrinsics</c>:
    /// X_cam→Y_world, Y_cam→Z_world (negated), Z_cam→X_world.</summary>
    public static readonly float[] RInit =
    [
        0f, 0f, 1f,
        1f, 0f, 0f,
        0f, -1f, 0f,
    ];

    /// <summary>Rigid-transform inverse: <c>[R|t]⁻¹ = [Rᵀ | −Rᵀ·t]</c>.</summary>
    public static void Invert(ReadOnlySpan<float> pose, Span<float> result)
    {
        ValidatePose(pose);
        ValidatePose(result);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                result[r * 4 + c] = pose[c * 4 + r];
        for (int r = 0; r < 3; r++)
        {
            float v = 0;
            for (int c = 0; c < 3; c++) v += result[r * 4 + c] * pose[c * 4 + 3];
            result[r * 4 + 3] = -v;
        }
        result[12] = 0f; result[13] = 0f; result[14] = 0f; result[15] = 1f;
    }

    /// <summary>Builds <c>Rz(yaw)·Ry(pitch)·Rx(roll)</c> from degrees into a row-major 3×3 (<c>float[9]</c>).</summary>
    public static void EulerDegreesToRotation(float rollDeg, float pitchDeg, float yawDeg, Span<float> rotation)
    {
        if (rotation.Length < 9) throw new ArgumentException("rotation must hold 9 floats.", nameof(rotation));
        float rx = rollDeg * MathF.PI / 180f, ry = pitchDeg * MathF.PI / 180f, rz = yawDeg * MathF.PI / 180f;
        float cx = MathF.Cos(rx), sx = MathF.Sin(rx);
        float cy = MathF.Cos(ry), sy = MathF.Sin(ry);
        float cz = MathF.Cos(rz), sz = MathF.Sin(rz);
        // Rz @ Ry @ Rx, row-major.
        rotation[0] = cz * cy; rotation[1] = cz * sy * sx - sz * cx; rotation[2] = cz * sy * cx + sz * sx;
        rotation[3] = sz * cy; rotation[4] = sz * sy * sx + cz * cx; rotation[5] = sz * sy * cx - cz * sx;
        rotation[6] = -sy; rotation[7] = cy * sx; rotation[8] = cy * cx;
    }

    /// <summary>Composes a camera-to-world pose from per-frame Euler rotation (degrees) and position, applying the
    /// Matrix-Game <see cref="RInit"/> axis remap and the 0.01 translation scale from <c>get_extrinsics</c>.</summary>
    public static void GetExtrinsics(float rollDeg, float pitchDeg, float yawDeg,
        float x, float y, float z, Span<float> pose)
    {
        ValidatePose(pose);
        Span<float> r = stackalloc float[9];
        EulerDegreesToRotation(rollDeg, pitchDeg, yawDeg, r);
        // R_world = R_init @ R.
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                float v = 0;
                for (int k = 0; k < 3; k++) v += RInit[i * 3 + k] * r[k * 3 + j];
                pose[i * 4 + j] = v;
            }
        pose[3] = x * 0.01f; pose[7] = y * 0.01f; pose[11] = z * 0.01f;
        pose[12] = 0f; pose[13] = 0f; pose[14] = 0f; pose[15] = 1f;
    }

    /// <summary>Identity pose.</summary>
    public static void Identity(Span<float> pose)
    {
        ValidatePose(pose);
        pose.Clear();
        pose[0] = 1f; pose[5] = 1f; pose[10] = 1f; pose[15] = 1f;
    }

    /// <summary>Spherical-linear interpolation between two unit quaternions <c>(x,y,z,w)</c>, taking the shortest
    /// path (sign-flips <paramref name="q1"/> when the dot product is negative).</summary>
    public static void Slerp(ReadOnlySpan<float> q0, ReadOnlySpan<float> q1, float t, Span<float> result)
    {
        if (q0.Length < 4 || q1.Length < 4 || result.Length < 4)
            throw new ArgumentException("quaternions must hold 4 floats.");
        float dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
        float sign = 1f;
        if (dot < 0f) { dot = -dot; sign = -1f; }
        float w0, w1;
        if (dot > 0.9995f)
        {
            w0 = 1f - t; w1 = t;   // nearly parallel: lerp
        }
        else
        {
            float theta = MathF.Acos(Math.Clamp(dot, -1f, 1f));
            float sinTheta = MathF.Sin(theta);
            w0 = MathF.Sin((1f - t) * theta) / sinTheta;
            w1 = MathF.Sin(t * theta) / sinTheta;
        }
        float nx = w0 * q0[0] + w1 * sign * q1[0];
        float ny = w0 * q0[1] + w1 * sign * q1[1];
        float nz = w0 * q0[2] + w1 * sign * q1[2];
        float nw = w0 * q0[3] + w1 * sign * q1[3];
        float inv = 1f / MathF.Sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
        result[0] = nx * inv; result[1] = ny * inv; result[2] = nz * inv; result[3] = nw * inv;
    }

    /// <summary>Integrates a Matrix-Game action stream into per-frame camera-to-world poses: W/S/A/D translate along
    /// the camera's local forward/right axes by <paramref name="moveStep"/>, mouse (Δx, Δy) accumulate into yaw/pitch
    /// (scaled by <paramref name="rotateScaleDeg"/> degrees per unit). Keyboard layout matches
    /// <c>MatrixGame3ActionEncoder</c> (W, S, A, D, extra, extra). Returns one pose per frame, continuing from
    /// <paramref name="startPose"/>'s position/heading. Step semantics are validation-gated.</summary>
    public static float[][] IntegrateActions(ReadOnlySpan<float> startPose,
        float[][] keyboardPerFrame, float[][] mousePerFrame, float moveStep, float rotateScaleDeg)
    {
        ValidatePose(startPose);
        if (keyboardPerFrame.Length != mousePerFrame.Length)
            throw new ArgumentException("keyboard and mouse streams must have the same frame count.");
        int frames = keyboardPerFrame.Length;

        // Recover heading from the start pose's forward axis (world Z column under the R_init remap).
        float yaw = MathF.Atan2(startPose[4], startPose[0]) * 180f / MathF.PI;
        float pitch = -MathF.Asin(Math.Clamp(startPose[8], -1f, 1f)) * 180f / MathF.PI;
        float px = startPose[3], py = startPose[7], pz = startPose[11];

        float[][] poses = new float[frames][];
        Span<float> rot = stackalloc float[9];
        for (int f = 0; f < frames; f++)
        {
            yaw += mousePerFrame[f][0] * rotateScaleDeg;
            pitch = Math.Clamp(pitch + mousePerFrame[f][1] * rotateScaleDeg, -89f, 89f);

            EulerDegreesToRotation(0f, pitch, yaw, rot);
            // Local forward = rotation's first row basis (x-forward convention pre-remap); right = second.
            float fwdX = rot[0], fwdY = rot[3], fwdZ = rot[6];
            float rightX = rot[1], rightY = rot[4], rightZ = rot[7];
            float[] kbd = keyboardPerFrame[f];
            float move = (kbd[0] - kbd[1]) * moveStep;      // W − S
            float strafe = (kbd[3] - kbd[2]) * moveStep;    // D − A
            px += fwdX * move + rightX * strafe;
            py += fwdY * move + rightY * strafe;
            pz += fwdZ * move + rightZ * strafe;

            float[] pose = new float[16];
            GetExtrinsics(0f, pitch, yaw, px / 0.01f, py / 0.01f, pz / 0.01f, pose);
            poses[f] = pose;
        }
        return poses;
    }

    private static void ValidatePose(ReadOnlySpan<float> pose)
    {
        if (pose.Length < 16) throw new ArgumentException("pose must hold 16 floats (row-major 4x4).");
    }
}
