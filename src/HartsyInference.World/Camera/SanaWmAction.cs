using System.Globalization;

namespace HartsyInference.World.Camera;

/// <summary>The SANA-WM action DSL (<c>--action</c>) and the smoothed velocity integrator that turns it into a camera path,
/// ported from upstream <c>camera_control.py</c> and <c>action_string_to_c2w</c> at <c>b3bad956</c>. A script is
/// comma-separated <c>&lt;keys&gt;-&lt;frames&gt;</c> segments; keys come from <c>wasdijkl</c> and combine (<c>dw</c>),
/// and <c>none</c> holds nothing. Poses are camera-to-world, row-major 4×4, OpenCV convention (+X right, +Y down,
/// +Z forward), starting from the identity. The integration runs in double precision and the poses are rounded to
/// single precision at the end, as upstream's float64 NumPy rollout and its <c>astype(np.float32)</c> do.</summary>
public static class SanaWmAction
{
    /// <summary>Frame rate the velocity smoothing is defined at.</summary>
    public const int Fps = 16;

    /// <summary>Translation per frame at full speed, in scene units.</summary>
    public const double TranslationPerFrame = 0.025;

    /// <summary>Yaw or pitch per frame at full speed, in degrees.</summary>
    public const double RotationDegreesPerFrame = 0.6;

    /// <summary>Pitch saturates at plus or minus this many degrees.</summary>
    public const double PitchLimitDegrees = 60.0;

    /// <summary>Smoothing time constant toward a non-zero target, in seconds.</summary>
    public const double TauPressSeconds = 0.45;

    /// <summary>Smoothing time constant toward rest, in seconds.</summary>
    public const double TauCoastSeconds = 1.0;

    /// <summary>The longest script, in frames, this parser expands.</summary>
    public const int MaxFrames = 65536;

    private const string AllowedKeys = "wasdijkl";
    private const double TargetEpsilon = 1e-12;
    private const double NormEpsilon = 1e-6;

    /// <summary>Expands <paramref name="action"/> into the controls held on each frame, one entry per generated frame.</summary>
    /// <exception cref="FormatException">The script is empty, a segment is malformed, a duration is not a positive
    /// integer, or a key is outside <c>wasdijkl</c>.</exception>
    public static SanaWmControls[] Parse(string action)
    {
        ArgumentNullException.ThrowIfNull(action);
        string cleaned = RemoveWhitespace(action.Replace('，', ','));
        if (cleaned.Length == 0)
            throw new FormatException("The action script is empty.");

        List<SanaWmControls> perFrame = [];
        foreach (string segment in cleaned.Split(','))
        {
            int dash = segment.LastIndexOf('-');
            if (segment.Length == 0 || dash < 0)
                throw new FormatException($"Invalid action segment '{segment}': expected '<keys>-<frames>'.");
            string keysPart = segment[..dash];
            string duration = segment[(dash + 1)..];
            if (duration.Length == 0 || !IsAsciiDigits(duration)
                || !int.TryParse(duration, NumberStyles.None, CultureInfo.InvariantCulture, out int frames) || frames <= 0)
            {
                throw new FormatException($"Action segment '{segment}' has a non-positive duration '{duration}'.");
            }
            if (perFrame.Count + (long)frames > MaxFrames)
                throw new FormatException($"The action script expands to more than {MaxFrames} frames.");
            SanaWmControls controls = ParseKeys(keysPart.ToLowerInvariant(), segment);
            for (int i = 0; i < frames; i++)
                perFrame.Add(controls);
        }
        return perFrame.ToArray();
    }

    /// <summary>Rolls <paramref name="action"/> out into <c>N + 1</c> camera-to-world poses (16 row-major floats each), the
    /// identity first and one pose per scripted frame after it.</summary>
    /// <exception cref="FormatException">The script does not parse; see <see cref="Parse"/>.</exception>
    public static float[][] ToCameraToWorld(string action) => Integrate(Parse(action));

    /// <summary>Integrates per-frame controls into <c>perFrame.Count + 1</c> camera-to-world poses with the smoothed velocity
    /// model: a newly pressed control snaps the velocity to its target; otherwise each component eases toward its target with
    /// <see cref="TauPressSeconds"/> (target non-zero) or <see cref="TauCoastSeconds"/> (target zero).</summary>
    public static float[][] Integrate(IReadOnlyList<SanaWmControls> perFrame)
    {
        ArgumentNullException.ThrowIfNull(perFrame);
        double rotationPerFrame = RotationDegreesPerFrame * (Math.PI / 180.0);
        double pitchLimit = PitchLimitDegrees * (Math.PI / 180.0);
        double dt = 1.0 / Fps;

        double[] pose = Identity();
        double pitch = 0.0;
        Velocity velocity = default;
        SanaWmControls last = SanaWmControls.None;
        float[][] poses = new float[perFrame.Count + 1][];
        poses[0] = ToSingle(pose);
        for (int f = 0; f < perFrame.Count; f++)
        {
            SanaWmControls controls = perFrame[f];
            Velocity target = Target(controls, rotationPerFrame);
            // A key pressed this frame takes effect at once; a held or released key eases toward its target.
            velocity = (controls & ~last) != SanaWmControls.None ? target : velocity.StepToward(target, dt);
            last = controls;
            pitch = Step(pose, pitch, pitchLimit, velocity);
            poses[f + 1] = ToSingle(pose);
        }
        return poses;
    }

    private static SanaWmControls ParseKeys(string keys, string segment)
    {
        if (keys == "none")
            return SanaWmControls.None;
        SanaWmControls controls = SanaWmControls.None;
        foreach (char key in keys)
        {
            controls |= key switch
            {
                'w' => SanaWmControls.Forward,
                's' => SanaWmControls.Back,
                'a' => SanaWmControls.YawLeft,
                'd' => SanaWmControls.YawRight,
                'i' => SanaWmControls.PitchUp,
                'k' => SanaWmControls.PitchDown,
                'j' => SanaWmControls.StrafeLeft,
                'l' => SanaWmControls.StrafeRight,
                _ => throw new FormatException($"Action segment '{segment}' contains the unknown key '{key}'; allowed: {AllowedKeys}."),
            };
        }
        return controls;
    }

    private static Velocity Target(SanaWmControls controls, double rotationPerFrame)
    {
        double forward = Axis(controls, SanaWmControls.Forward, SanaWmControls.Back);
        double strafe = Axis(controls, SanaWmControls.StrafeRight, SanaWmControls.StrafeLeft);
        double yaw = Axis(controls, SanaWmControls.YawRight, SanaWmControls.YawLeft);
        double pitch = Axis(controls, SanaWmControls.PitchUp, SanaWmControls.PitchDown);
        return new Velocity(forward * TranslationPerFrame, strafe * TranslationPerFrame, yaw * rotationPerFrame,
            pitch * rotationPerFrame);
    }

    private static double Axis(SanaWmControls controls, SanaWmControls positive, SanaWmControls negative) =>
        ((controls & positive) != 0 ? 1.0 : 0.0) - ((controls & negative) != 0 ? 1.0 : 0.0);

    // CameraPoseIntegrator.step: R' = rot_y(yaw)·R·rot_x(pitch step); translation on the y = 0 plane along R' forward and right.
    private static double Step(double[] pose, double pitch, double pitchLimit, Velocity v)
    {
        double newPitch = Math.Max(-pitchLimit, Math.Min(pitchLimit, pitch + v.Pitch));
        double pitchStep = newPitch - pitch;

        Span<double> rotation = stackalloc double[9];
        Span<double> yawed = stackalloc double[9];
        Span<double> next = stackalloc double[9];
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
                rotation[r * 3 + c] = pose[r * 4 + c];
        }
        Multiply(RotY(v.Yaw), rotation, yawed);
        Multiply(yawed, RotX(pitchStep), next);

        (double fx, double fz) = Horizontal(next[2], next[8]);
        (double rx, double rz) = Horizontal(next[0], next[6]);
        double tx = pose[3] + fx * v.Forward + rx * v.Strafe;
        double tz = pose[11] + fz * v.Forward + rz * v.Strafe;
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
                pose[r * 4 + c] = next[r * 3 + c];
        }
        pose[3] = tx;
        pose[11] = tz;
        return newPitch;
    }

    // Projects a column onto the y = 0 plane and normalizes it with upstream's 1e-6 guard in the divisor.
    private static (double X, double Z) Horizontal(double x, double z)
    {
        double norm = Math.Sqrt(x * x + z * z);
        return norm > 0 ? (x / (norm + NormEpsilon), z / (norm + NormEpsilon)) : (x, z);
    }

    private static double[] RotX(double angle)
    {
        double c = Math.Cos(angle), s = Math.Sin(angle);
        return [1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c];
    }

    private static double[] RotY(double angle)
    {
        double c = Math.Cos(angle), s = Math.Sin(angle);
        return [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c];
    }

    private static void Multiply(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
    {
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
                result[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
        }
    }

    private static double[] Identity() => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

    private static float[] ToSingle(double[] pose)
    {
        float[] result = new float[16];
        for (int i = 0; i < 16; i++)
            result[i] = (float)pose[i];
        return result;
    }

    private static string RemoveWhitespace(string text)
    {
        System.Text.StringBuilder builder = new(text.Length);
        foreach (char c in text)
        {
            if (!char.IsWhiteSpace(c))
                builder.Append(c);
        }
        return builder.ToString();
    }

    private static bool IsAsciiDigits(string text)
    {
        foreach (char c in text)
        {
            if (!char.IsAsciiDigit(c))
                return false;
        }
        return true;
    }

    /// <summary>Per-frame velocity: forward and strafe-right translation, yaw-right and pitch-up rotation.</summary>
    private readonly record struct Velocity(double Forward, double Strafe, double Yaw, double Pitch)
    {
        public Velocity StepToward(Velocity target, double dt) => new(
            Ease(Forward, target.Forward, dt), Ease(Strafe, target.Strafe, dt), Ease(Yaw, target.Yaw, dt), Ease(Pitch, target.Pitch, dt));

        private static double Ease(double current, double target, double dt)
        {
            double tau = Math.Abs(target) > TargetEpsilon ? TauPressSeconds : TauCoastSeconds;
            double alpha = 1.0 - Math.Exp(-dt / tau);
            return current + alpha * (target - current);
        }
    }
}
