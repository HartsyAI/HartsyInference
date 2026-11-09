namespace HartsyInference.ThreeD.Geometry;

/// <summary>
/// Static spring-chain preview (hairdeck.infer.glb.chain_trimesh): an 8-sided tube per joint segment and an icosphere
/// (one subdivision, 1.9 times the tube radius) per joint. Tube vertices are merged as trimesh's constructor does.
/// </summary>
public static class ChainTubeMesher
{
    /// <summary>Tube radius in metres, as the reference.</summary>
    public const double DefaultRadius = 0.0022;

    private const int Sections = 8;
    private const double JointRadiusScale = 1.9;
    private const double MinSegmentLength = 1e-6;
    private const double IcosahedronComponent = 0.85065080835204;
    private const double IcosahedronRing = 0.5257311121191336;

    private static readonly Vector3d[] IcosahedronVertices =
    [
        new(-IcosahedronRing, IcosahedronComponent, 0), new(IcosahedronRing, IcosahedronComponent, 0),
        new(-IcosahedronRing, -IcosahedronComponent, 0), new(IcosahedronRing, -IcosahedronComponent, 0),
        new(0, -IcosahedronRing, IcosahedronComponent), new(0, IcosahedronRing, IcosahedronComponent),
        new(0, -IcosahedronRing, -IcosahedronComponent), new(0, IcosahedronRing, -IcosahedronComponent),
        new(IcosahedronComponent, 0, -IcosahedronRing), new(IcosahedronComponent, 0, IcosahedronRing),
        new(-IcosahedronComponent, 0, -IcosahedronRing), new(-IcosahedronComponent, 0, IcosahedronRing),
    ];

    private static readonly int[][] IcosahedronFaces =
    [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
        [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
        [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];

    /// <summary>Meshes the joint polylines (M joints each, xyz metres) of all chains into one mesh; null when there is nothing to draw.</summary>
    public static Mesh? Build(IReadOnlyList<float[]> chains, double radius = DefaultRadius)
    {
        ArgumentNullException.ThrowIfNull(chains);
        if (!(radius > 0) || !double.IsFinite(radius))
            throw new ArgumentOutOfRangeException(nameof(radius), "The tube radius must be positive and finite.");

        Builder builder = new();
        foreach (float[] chain in chains)
        {
            ArgumentNullException.ThrowIfNull(chain);
            if (chain.Length % 3 != 0)
                throw new ArgumentException("A chain's joints must be xyz triples.", nameof(chains));
            Vector3d[] joints = new Vector3d[chain.Length / 3];
            for (int j = 0; j < joints.Length; j++)
                joints[j] = new Vector3d(chain[3 * j], chain[3 * j + 1], chain[3 * j + 2]);
            for (int j = 0; j + 1 < joints.Length; j++)
                if ((joints[j + 1] - joints[j]).Length > MinSegmentLength)
                    builder.AddTube(joints[j], joints[j + 1], radius);
            foreach (Vector3d joint in joints)
                builder.AddSphere(joint, JointRadiusScale * radius);
        }
        return builder.ToMesh();
    }

    private sealed class Builder
    {
        private readonly List<float> _positions = [];
        private readonly List<int> _indices = [];
        private int _vertexCount;

        public void AddTube(Vector3d a, Vector3d b, double radius)
        {
            Vector3d vector = b - a;
            double half = vector.Length / 2.0;
            Vector3d mid = a + vector * 0.5;
            double[] rotation = AlignZ(vector);
            int first = _vertexCount;
            int bottomAxis = first;
            int topAxis = first + 1 + 2 * Sections;
            AddVertex(mid + Rotate(rotation, new Vector3d(0, 0, -half)));
            Vector3d[] bottom = new Vector3d[Sections];
            Vector3d[] top = new Vector3d[Sections];
            for (int s = 0; s < Sections; s++)
            {
                double theta = s * (2.0 * Math.PI / Sections);
                double x = Math.Cos(theta) * radius;
                double y = Math.Sin(theta) * radius;
                bottom[s] = mid + Rotate(rotation, new Vector3d(x, y, -half));
                top[s] = mid + Rotate(rotation, new Vector3d(x, y, half));
            }
            for (int s = 0; s < Sections; s++)
                AddVertex(bottom[s]);
            for (int s = 0; s < Sections; s++)
                AddVertex(top[s]);
            AddVertex(mid + Rotate(rotation, new Vector3d(0, 0, half)));

            for (int s = 0; s < Sections; s++)
            {
                int t = (s + 1) % Sections;
                int b0 = first + 1 + s;
                int b1 = first + 1 + t;
                int t0 = first + 1 + Sections + s;
                int t1 = first + 1 + Sections + t;
                AddTriangle(b0, bottomAxis, b1);
                AddTriangle(b0, b1, t0);
                AddTriangle(t0, b1, t1);
                AddTriangle(t0, t1, topAxis);
            }
        }

        public void AddSphere(Vector3d center, double radius)
        {
            List<Vector3d> vertices = [.. IcosahedronVertices];
            List<int[]> faces = [];
            Dictionary<(int, int), int> midpoints = [];
            foreach (int[] face in IcosahedronFaces)
            {
                int ab = Midpoint(vertices, midpoints, face[0], face[1]);
                int bc = Midpoint(vertices, midpoints, face[1], face[2]);
                int ca = Midpoint(vertices, midpoints, face[2], face[0]);
                faces.Add([face[0], ab, ca]);
                faces.Add([ab, face[1], bc]);
                faces.Add([ca, bc, face[2]]);
                faces.Add([ab, bc, ca]);
            }
            int first = _vertexCount;
            foreach (Vector3d v in vertices)
            {
                double scalar = v.Length;
                Vector3d onSphere = v + (v / scalar) * (radius - scalar);
                AddVertex(onSphere + center);
            }
            foreach (int[] face in faces)
                AddTriangle(first + face[0], first + face[1], first + face[2]);
        }

        public Mesh? ToMesh()
        {
            if (_indices.Count == 0)
                return null;
            return new Mesh { Vertices = _positions.ToArray(), Indices = _indices.ToArray() };
        }

        private static int Midpoint(List<Vector3d> vertices, Dictionary<(int, int), int> midpoints, int a, int b)
        {
            (int, int) key = a < b ? (a, b) : (b, a);
            if (midpoints.TryGetValue(key, out int index))
                return index;
            index = vertices.Count;
            vertices.Add((vertices[a] + vertices[b]) / 2.0);
            midpoints[key] = index;
            return index;
        }

        private void AddVertex(Vector3d v)
        {
            _positions.Add((float)v.X);
            _positions.Add((float)v.Y);
            _positions.Add((float)v.Z);
            _vertexCount++;
        }

        private void AddTriangle(int a, int b, int c)
        {
            _indices.Add(a);
            _indices.Add(b);
            _indices.Add(c);
        }
    }

    /// <summary>
    /// Rotation taking the z axis to <paramref name="direction"/> as trimesh.geometry.align_vectors builds it: the
    /// orthogonal bases of both vectors (SVD columns, reproduced with LAPACK's Householder reflector) multiplied.
    /// </summary>
    private static double[] AlignZ(Vector3d direction)
    {
        double[] from = SvdBasis(new Vector3d(0, 0, 1));
        double[] to = SvdBasis(direction);
        double[] rotation = new double[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                for (int k = 0; k < 3; k++)
                    rotation[3 * r + c] += to[3 * r + k] * from[3 * c + k];
        return rotation;
    }

    /// <summary>The U factor of np.linalg.svd of a column vector: a Householder reflector, column 0 signed, det +1.</summary>
    private static double[] SvdBasis(Vector3d x)
    {
        double alpha = x.X;
        double xNorm = double.Hypot(x.Y, x.Z);
        double beta = alpha;
        double tau = 0.0;
        Vector3d v = new(1, 0, 0);
        if (xNorm != 0.0)
        {
            beta = -Math.CopySign(double.Hypot(alpha, xNorm), alpha);
            tau = (beta - alpha) / beta;
            v = new Vector3d(1, x.Y / (alpha - beta), x.Z / (alpha - beta));
        }
        double[] vv = [v.X, v.Y, v.Z];
        double sign = Math.Sign(beta);
        double[] u = new double[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                u[3 * r + c] = ((r == c ? 1.0 : 0.0) - tau * (vv[r] * vv[c])) * (c == 0 ? sign : 1.0);
        double determinant = u[0] * (u[4] * u[8] - u[5] * u[7]) - u[1] * (u[3] * u[8] - u[5] * u[6])
            + u[2] * (u[3] * u[7] - u[4] * u[6]);
        if (determinant < 0)
            for (int r = 0; r < 3; r++)
                u[3 * r + 2] = -u[3 * r + 2];
        return u;
    }

    private static Vector3d Rotate(double[] m, Vector3d v) =>
        new(m[0] * v.X + m[1] * v.Y + m[2] * v.Z,
            m[3] * v.X + m[4] * v.Y + m[5] * v.Z,
            m[6] * v.X + m[7] * v.Y + m[8] * v.Z);
}
