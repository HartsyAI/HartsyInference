namespace HartsyInference.ThreeD.Geometry.HairRig;

/// <summary>
/// Spring-bone simulation of hair chains and the four head motions of the reference, ported from HairDeck's
/// <c>hairdeck/spring/sim.py</c> (Apache-2.0). The chain update follows UniVRM's <c>VRMSpringBoneLogic</c> (MIT, Copyright
/// (c) 2020 VRM Consortium): verlet with drag, a stiffness pull toward the parent-rotated rest direction, gravity and a
/// length constraint, at 60 fps in metres. Matrices are row-major 3x3 stored in <c>double[9]</c>; vectors are <c>double[3]</c>.
/// </summary>
public static class SpringBoneSim
{
    /// <summary>Simulation rate of the reference (frames per second).</summary>
    public const int Fps = 60;

    /// <summary>Frames of each motion (2.5 s at <see cref="Fps"/>).</summary>
    public const int Frames = 150;

    /// <summary>Motion names in the reference's order.</summary>
    public static readonly string[] MotionNames = ["turn", "nod", "jump", "sway"];

    private const double Dt = 1.0 / Fps;

    /// <summary>One chain: joints in metres (at least two, no zero-length segment), the stiffness of the pull toward the rest
    /// direction, the drag and the gravity acceleration.</summary>
    public sealed record Chain(double[] Joints, double Stiffness, double Drag, double[] Gravity);

    /// <summary>Head rotation (<see cref="Frames"/> row-major 3x3, frame-major) and translation (<see cref="Frames"/> xyz) of one motion.</summary>
    public sealed record Motion(double[] Rotations, double[] Translations);

    /// <summary>Simulated bones of one chain: head positions (frame, bone, xyz) and rest-to-current rotations (frame, bone, 3x3).</summary>
    public sealed record ChainState(double[] Heads, double[] Rotations, int Bones);

    /// <summary>The head motion <paramref name="name"/> (turn, nod, jump or sway) about the head pivot, in the head frame (y up, face toward +z).</summary>
    public static Motion GetMotion(string name)
    {
        double[] rot = new double[Frames * 9];
        double[] trans = new double[Frames * 3];
        for (int f = 0; f < Frames; f++)
        {
            SetIdentity(rot, f * 9);
        }

        for (int f = 0; f < Frames; f++)
        {
            double t = (f + 1) / (double)Fps;
            switch (name)
            {
                case "turn":
                {
                    double x = Math.Clamp(t / 0.35, 0.0, 1.0);
                    WriteRotation(rot, f * 9, RotY(Deg(50) * x * x * (3 - 2 * x)));
                    break;
                }
                case "nod":
                {
                    double angle = t < 0.5 ? Deg(25) * Math.Sin(Math.PI * t / 0.5) : 0.0;
                    WriteRotation(rot, f * 9, RotX(angle));
                    break;
                }
                case "jump":
                    trans[f * 3 + 1] = t < 0.6 ? 0.15 * 4 * (t / 0.6) * (1 - t / 0.6) : 0.0;
                    break;
                case "sway":
                    trans[f * 3] = t < 4.0 / 3.0 ? 0.06 * Math.Sin(2 * Math.PI * 1.5 * t) : 0.0;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(name), name, "Unknown motion; expected turn, nod, jump or sway.");
            }
        }

        return new Motion(rot, trans);
    }

    /// <summary>Simulates every chain under <paramref name="motion"/> about <paramref name="pivot"/>.</summary>
    public static ChainState[] Simulate(IReadOnlyList<Chain> chains, Motion motion, double[] pivot)
    {
        ArgumentNullException.ThrowIfNull(chains);
        ArgumentNullException.ThrowIfNull(motion);
        ArgumentNullException.ThrowIfNull(pivot);
        ChainState[] states = new ChainState[chains.Count];
        for (int c = 0; c < chains.Count; c++)
        {
            states[c] = SimulateChain(chains[c], motion, pivot);
        }

        return states;
    }

    private static ChainState SimulateChain(Chain chain, Motion motion, double[] pivot)
    {
        double[] joints = chain.Joints;
        int n = joints.Length / 3;
        if (joints.Length % 3 != 0 || n < 2)
        {
            throw new ArgumentException("A chain needs at least two joints (xyz triples).", nameof(chain));
        }

        int bones = n - 1;
        double[] restDir = new double[bones * 3];
        double[] length = new double[bones];
        for (int k = 0; k < bones; k++)
        {
            double dx = joints[3 * (k + 1)] - joints[3 * k];
            double dy = joints[3 * (k + 1) + 1] - joints[3 * k + 1];
            double dz = joints[3 * (k + 1) + 2] - joints[3 * k + 2];
            double len = Math.Sqrt(dx * dx + dy * dy + dz * dz);
            length[k] = len;
            restDir[3 * k] = dx / len;
            restDir[3 * k + 1] = dy / len;
            restDir[3 * k + 2] = dz / len;
        }

        // cur and prev hold the current and previous bone tips (the joints 1..n-1), as in sim.py.
        double[] cur = new double[bones * 3];
        Array.Copy(joints, 3, cur, 0, bones * 3);
        double[] prev = (double[])cur.Clone();
        double[] heads = new double[Frames * bones * 3];
        double[] rots = new double[Frames * bones * 9];
        double[] head = new double[3];
        double[] rest = new double[3];
        double[] nxt = new double[3];
        double[] v = new double[3];
        double[] wp = new double[9];
        double[] wk = new double[9];
        double[] root = [joints[0], joints[1], joints[2]];

        for (int f = 0; f < Frames; f++)
        {
            Array.Copy(motion.Rotations, f * 9, wp, 0, 9);
            double[] p0 = Sub(root, pivot);
            MulMV(wp, p0, head);
            for (int i = 0; i < 3; i++)
            {
                head[i] += pivot[i] + motion.Translations[f * 3 + i];
            }

            for (int k = 0; k < bones; k++)
            {
                MulMV(wp, [restDir[3 * k], restDir[3 * k + 1], restDir[3 * k + 2]], rest);
                for (int i = 0; i < 3; i++)
                {
                    nxt[i] = cur[3 * k + i] + (cur[3 * k + i] - prev[3 * k + i]) * (1 - chain.Drag)
                        + rest[i] * (chain.Stiffness * Dt) + chain.Gravity[i] * Dt;
                    v[i] = nxt[i] - head[i];
                }

                double norm = Math.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                if (norm > 1e-12)
                {
                    for (int i = 0; i < 3; i++)
                    {
                        v[i] /= norm;
                    }
                }
                else
                {
                    Array.Copy(rest, v, 3);
                }

                for (int i = 0; i < 3; i++)
                {
                    nxt[i] = head[i] + v[i] * length[k];
                    prev[3 * k + i] = cur[3 * k + i];
                    cur[3 * k + i] = nxt[i];
                }

                RotDiff(rest, v, wk);
                MulMM(wk, wp, wk);
                Array.Copy(head, 0, heads, (f * bones + k) * 3, 3);
                Array.Copy(wk, 0, rots, (f * bones + k) * 9, 9);
                Array.Copy(wk, wp, 9);
                Array.Copy(nxt, head, 3);
            }
        }

        return new ChainState(heads, rots, bones);
    }

    /// <summary>Rotation matrix (row-major, 9 values) that takes unit vector <paramref name="a"/> to unit vector <paramref name="b"/>
    /// along the shortest arc; identity when they are parallel.</summary>
    public static void RotDiff(double[] a, double[] b, double[] result)
    {
        double vx = a[1] * b[2] - a[2] * b[1];
        double vy = a[2] * b[0] - a[0] * b[2];
        double vz = a[0] * b[1] - a[1] * b[0];
        double s = Math.Sqrt(vx * vx + vy * vy + vz * vz);
        double c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        if (s < 1e-12)
        {
            SetIdentity(result, 0);
            return;
        }

        double kx = vx / s, ky = vy / s, kz = vz / s;
        double[] kk = [0, -kz, ky, kz, 0, -kx, -ky, kx, 0];
        double angle = Math.Atan2(s, c);
        double sin = Math.Sin(angle);
        double oneMinusCos = 1 - Math.Cos(angle);
        double[] kk2 = new double[9];
        MulMM(kk, kk, kk2);
        for (int i = 0; i < 9; i++)
        {
            result[i] = (i % 4 == 0 ? 1.0 : 0.0) + sin * kk[i] + oneMinusCos * kk2[i];
        }
    }

    /// <summary>Row-major 3x3 product <c>a * b</c> written to <paramref name="result"/> (which must not alias <paramref name="b"/>
    /// unless it is <paramref name="a"/>; the product is taken through a temporary).</summary>
    public static void MulMM(double[] a, double[] b, double[] result)
    {
        double[] t = new double[9];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                t[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
            }
        }

        Array.Copy(t, result, 9);
    }

    /// <summary>Row-major 3x3 times vector.</summary>
    public static void MulMV(double[] m, double[] v, double[] result)
    {
        double x = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
        double y = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
        double z = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
        result[0] = x;
        result[1] = y;
        result[2] = z;
    }

    private static double Deg(double degrees) => degrees * Math.PI / 180.0;

    private static double[] RotX(double a)
    {
        double c = Math.Cos(a), s = Math.Sin(a);
        return [1, 0, 0, 0, c, -s, 0, s, c];
    }

    private static double[] RotY(double a)
    {
        double c = Math.Cos(a), s = Math.Sin(a);
        return [c, 0, s, 0, 1, 0, -s, 0, c];
    }

    private static void WriteRotation(double[] dst, int offset, double[] m) => Array.Copy(m, 0, dst, offset, 9);

    private static void SetIdentity(double[] m, int offset)
    {
        for (int i = 0; i < 9; i++)
        {
            m[offset + i] = i % 4 == 0 ? 1.0 : 0.0;
        }
    }

    private static double[] Sub(double[] a, double[] b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
