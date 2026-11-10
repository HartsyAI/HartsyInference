using System.Globalization;

namespace HartsyInference.World.Camera;

/// <summary>The SANA-WM action DSL (<c>--action</c>) and its rollout into a camera path through
/// <see cref="SanaWmCameraIntegrator"/>, ported from upstream <c>camera_control.py</c> and <c>action_string_to_c2w</c> at
/// <c>b3bad956</c>. A script is
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
    /// model of <see cref="SanaWmCameraIntegrator"/>: the identity, then one pose per frame.</summary>
    public static float[][] Integrate(IReadOnlyList<SanaWmControls> perFrame)
    {
        ArgumentNullException.ThrowIfNull(perFrame);
        SanaWmCameraIntegrator integrator = new();
        float[][] poses = new float[perFrame.Count + 1][];
        poses[0] = integrator.Pose;
        for (int f = 0; f < perFrame.Count; f++)
            poses[f + 1] = integrator.Step(perFrame[f]);
        return poses;
    }

    /// <summary>The controls a key string names: letters from <c>wasdijkl</c> in any order and case, or <c>none</c> (or
    /// nothing) for no key held.</summary>
    /// <exception cref="FormatException">A key is outside <c>wasdijkl</c>.</exception>
    public static SanaWmControls ParseKeys(string keys)
    {
        ArgumentNullException.ThrowIfNull(keys);
        string cleaned = RemoveWhitespace(keys).ToLowerInvariant();
        return cleaned.Length == 0 ? SanaWmControls.None : ParseKeys(cleaned, cleaned);
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
}
