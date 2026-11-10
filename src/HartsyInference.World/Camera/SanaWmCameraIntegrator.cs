namespace HartsyInference.World.Camera;

/// <summary>SANA-WM's smoothed velocity camera model, one frame per <see cref="Step"/> (upstream
/// <c>camera_control.CameraPoseIntegrator</c> with <c>VelocityState</c>): a newly pressed control snaps the velocity to its
/// target; otherwise each component eases toward its target with <see cref="SanaWmAction.TauPressSeconds"/> (target
/// non-zero) or <see cref="SanaWmAction.TauCoastSeconds"/> (target zero). The pose starts at the identity. The state is in
/// double precision and each returned pose is rounded to single, as upstream's float64 rollout and its
/// <c>astype(np.float32)</c> do. <see cref="SanaWmAction.Integrate"/> is this integrator run over a whole script.</summary>
public sealed class SanaWmCameraIntegrator : ICameraIntegrator<SanaWmControls>
{
    private const double TargetEpsilon = 1e-12;
    private const double NormEpsilon = 1e-6;
    private const double RotationPerFrame = SanaWmAction.RotationDegreesPerFrame * (Math.PI / 180.0);
    private const double PitchLimit = SanaWmAction.PitchLimitDegrees * (Math.PI / 180.0);
    private const double FrameSeconds = 1.0 / SanaWmAction.Fps;

    private readonly double[] _pose = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    private double _pitch;
    private Velocity _velocity;
    private SanaWmControls _last = SanaWmControls.None;

    /// <summary>Frames integrated so far.</summary>
    public int Frames { get; private set; }

    /// <inheritdoc/>
    public float[] Pose
    {
        get
        {
            float[] result = new float[16];
            for (int i = 0; i < 16; i++)
                result[i] = (float)_pose[i];
            return result;
        }
    }

    /// <inheritdoc/>
    public float[] Step(SanaWmControls controls)
    {
        Velocity target = Target(controls);
        // A key pressed this frame takes effect at once; a held or released key eases toward its target.
        _velocity = (controls & ~_last) != SanaWmControls.None ? target : _velocity.StepToward(target, FrameSeconds);
        _last = controls;
        _pitch = Advance(_pose, _pitch, _velocity);
        Frames++;
        return Pose;
    }

    private static Velocity Target(SanaWmControls controls)
    {
        double forward = Axis(controls, SanaWmControls.Forward, SanaWmControls.Back);
        double strafe = Axis(controls, SanaWmControls.StrafeRight, SanaWmControls.StrafeLeft);
        double yaw = Axis(controls, SanaWmControls.YawRight, SanaWmControls.YawLeft);
        double pitch = Axis(controls, SanaWmControls.PitchUp, SanaWmControls.PitchDown);
        return new Velocity(forward * SanaWmAction.TranslationPerFrame, strafe * SanaWmAction.TranslationPerFrame,
            yaw * RotationPerFrame, pitch * RotationPerFrame);
    }

    private static double Axis(SanaWmControls controls, SanaWmControls positive, SanaWmControls negative) =>
        ((controls & positive) != 0 ? 1.0 : 0.0) - ((controls & negative) != 0 ? 1.0 : 0.0);

    // CameraPoseIntegrator.step: R' = rot_y(yaw)·R·rot_x(pitch step); translation on the y = 0 plane along R' forward and right.
    private static double Advance(double[] pose, double pitch, Velocity v)
    {
        double newPitch = Math.Max(-PitchLimit, Math.Min(PitchLimit, pitch + v.Pitch));
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

    /// <summary>Per-frame velocity: forward and strafe-right translation, yaw-right and pitch-up rotation.</summary>
    private readonly record struct Velocity(double Forward, double Strafe, double Yaw, double Pitch)
    {
        public Velocity StepToward(Velocity target, double dt) => new(
            Ease(Forward, target.Forward, dt), Ease(Strafe, target.Strafe, dt), Ease(Yaw, target.Yaw, dt), Ease(Pitch, target.Pitch, dt));

        private static double Ease(double current, double target, double dt)
        {
            double tau = Math.Abs(target) > TargetEpsilon ? SanaWmAction.TauPressSeconds : SanaWmAction.TauCoastSeconds;
            double alpha = 1.0 - Math.Exp(-dt / tau);
            return current + alpha * (target - current);
        }
    }
}
