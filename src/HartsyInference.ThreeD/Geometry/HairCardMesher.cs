namespace HartsyInference.ThreeD.Geometry;

/// <summary>
/// Port of hairdeck.cards.mesh (Bishop-frame card ribbons, card UVs, colours along a card) and hairdeck.infer.glb
/// (hair_trimesh: all cards as one mesh with per-card brightness). Cards with fewer than three points are skipped.
/// </summary>
public static class HairCardMesher
{
    private const int MinPoints = 3;
    private const int RowStride = 8;
    private const double ParallelTolerance = 1e-9;
    private const double BrightnessMean = 0.93;
    private const double BrightnessDeviation = 0.07;
    private const double BrightnessMin = 0.78;
    private const double BrightnessMax = 1.0;
    private const double DegenerateFaceNorm = 1e-14;
    private const double LinearSrgbThreshold = 0.0031308;
    private const double SrgbThreshold = 0.04045;
    private static readonly double[] Stencil = [-1.0 / 12, 2.0 / 3, 0.0, -2.0 / 3, 1.0 / 12];
    private static readonly double[] UCoordinates = [0.0, 0.5, 1.0, 0.5];

    /// <summary>
    /// Meshes <paramref name="cards"/> in order. Colours are either on every card or on none; with colours the base colour
    /// is white and COLOR_0 carries them, otherwise COLOR_0 is the brightness alone and the base colour is
    /// <paramref name="hairColorSrgb"/>. Returns null when no card has a triangle.
    /// </summary>
    public static HairCardMesh? Build(IReadOnlyList<HairCard> cards, float[] hairColorSrgb)
    {
        ArgumentNullException.ThrowIfNull(cards);
        ArgumentNullException.ThrowIfNull(hairColorSrgb);
        if (hairColorSrgb.Length != 3)
            throw new ArgumentException("The hair colour must be three sRGB values.", nameof(hairColorSrgb));
        bool cardColors = cards.Count > 0 && cards[0].Colors is not null;
        ValidateCards(cards, cardColors);

        NumpyNormalGenerator brightness = new(0);
        List<float> positions = [];
        List<int> indices = [];
        List<float> uvs = [];
        List<float> colors = [];
        int offset = 0;
        foreach (HairCard card in cards)
        {
            int n = card.Sequence.Length / RowStride;
            if (n < MinPoints)
                continue;
            double[] vertices = CardVertices(card.Sequence, n, card.ShapeClass == 1);
            double[] vCoordinates = CardVCoordinates(card.Sequence, n);
            double factor = Math.Clamp(brightness.Normal(BrightnessMean, BrightnessDeviation), BrightnessMin, BrightnessMax);
            double[]? srgb = cardColors ? SrgbAlongCard(card.Colors!, vCoordinates) : null;

            foreach (int face in CardFaces(vertices, n))
                indices.Add(face + offset);
            for (int v = 0; v < 4 * n; v++)
            {
                positions.Add((float)vertices[3 * v]);
                positions.Add((float)vertices[3 * v + 1]);
                positions.Add((float)vertices[3 * v + 2]);
                uvs.Add((float)UCoordinates[v % 4]);
                uvs.Add((float)vCoordinates[v]);
                for (int c = 0; c < 3; c++)
                {
                    double linear = srgb is null ? 1.0 : SrgbToLinear(Math.Clamp(srgb[3 * v + c], 0.0, 1.0));
                    colors.Add((float)(factor * linear));
                }
            }
            offset += 4 * n;
        }
        if (offset == 0 || indices.Count == 0)
            return null;

        float[] baseColor = cardColors
            ? [1f, 1f, 1f, 1f]
            : [(float)SrgbToLinear(hairColorSrgb[0]), (float)SrgbToLinear(hairColorSrgb[1]),
               (float)SrgbToLinear(hairColorSrgb[2]), 1f];
        Mesh mesh = new()
        {
            Vertices = positions.ToArray(),
            Indices = indices.ToArray(),
            Uvs = uvs.ToArray(),
            VertexColors = colors.ToArray(),
        };
        return new HairCardMesh(mesh, baseColor);
    }

    /// <summary>Vertices of one card's ribbon (4 per control point, reference layout), float64, row-major xyz.</summary>
    internal static double[] CardVertices(double[] sequence, int n, bool triangle)
    {
        Vector3d[] points = new Vector3d[n];
        for (int i = 0; i < n; i++)
            points[i] = new Vector3d(sequence[RowStride * i], sequence[RowStride * i + 1], sequence[RowStride * i + 2]);
        Vector3d[] tangents = Tangents(points);

        Vector3d[] normals = new Vector3d[n];
        Vector3d root = new(sequence[5], sequence[6], sequence[7]);
        normals[0] = Vector3d.Unit(root - Vector3d.Dot(root, tangents[0]) * tangents[0]);
        for (int i = 1; i < n; i++)
        {
            Vector3d a = tangents[i - 1];
            Vector3d b = tangents[i];
            Vector3d axis = Vector3d.Cross(a, b);
            double sin = axis.Length;
            double cos = Math.Clamp(Vector3d.Dot(a, b), -1.0, 1.0);
            Vector3d m = normals[i - 1];
            Vector3d k = axis / (sin > ParallelTolerance ? sin : 1.0);
            Vector3d rotated = m * cos + Vector3d.Cross(k, m) * sin + (k * Vector3d.Dot(k, m)) * (1.0 - cos);
            m = sin > ParallelTolerance ? rotated : m;
            normals[i] = Vector3d.Unit(m - Vector3d.Dot(m, b) * b);
        }

        double up = triangle ? 3.0 : 2.0;
        double down = triangle ? 6.0 : 2.0;
        double[] vertices = new double[12 * n];
        for (int i = 0; i < n; i++)
        {
            Vector3d normal = Vector3d.Unit(normals[i]);
            Vector3d tangent = Vector3d.Unit(tangents[i] - Vector3d.Dot(tangents[i], normal) * normal);
            Vector3d binormal = Vector3d.Cross(normal, tangent);
            double width = sequence[RowStride * i + 3];
            double thickness = sequence[RowStride * i + 4];
            Vector3d p = points[i];
            Vector3d[] ring =
            [
                p + width / 2 * normal,
                p + (thickness / up) * binormal,
                p - width / 2 * normal,
                p - (thickness / down) * binormal,
            ];
            for (int k = 0; k < 4; k++)
            {
                vertices[12 * i + 3 * k] = ring[k].X;
                vertices[12 * i + 3 * k + 1] = ring[k].Y;
                vertices[12 * i + 3 * k + 2] = ring[k].Z;
            }
        }
        return vertices;
    }

    /// <summary>Triangle indices of one card's ribbon, local to the card, with zero-area faces dropped.</summary>
    internal static List<int> CardFaces(double[] vertices, int n)
    {
        List<int> faces = [];
        foreach (int[] f in RingFaces(n))
        {
            Vector3d a = At(vertices, f[0]);
            Vector3d b = At(vertices, f[1]);
            Vector3d c = At(vertices, f[2]);
            Vector3d area = Vector3d.Cross(b - a, c - a);
            if (area.Length > DegenerateFaceNorm)
                faces.AddRange(f);
        }
        return faces;
    }

    /// <summary>Per-vertex v of card_uv: arc length from the root over the card length (4 entries per control point).</summary>
    internal static double[] CardVCoordinates(double[] sequence, int n)
    {
        double[] arc = new double[n];
        for (int i = 1; i < n; i++)
        {
            Vector3d d = new(sequence[RowStride * i] - sequence[RowStride * (i - 1)],
                sequence[RowStride * i + 1] - sequence[RowStride * (i - 1) + 1],
                sequence[RowStride * i + 2] - sequence[RowStride * (i - 1) + 2]);
            arc[i] = arc[i - 1] + d.Length;
        }
        double total = arc[n - 1];
        double[] v = new double[4 * n];
        for (int i = 0; i < n; i++)
        {
            double value = total > 0 ? arc[i] / total : i * (1.0 / (n - 1));
            v[4 * i] = value;
            v[4 * i + 1] = value;
            v[4 * i + 2] = value;
            v[4 * i + 3] = value;
        }
        return v;
    }

    /// <summary>Ring faces of hairdeck.cards.mesh.ring_faces for n control points (two root caps and ten faces per ring).</summary>
    internal static List<int[]> RingFaces(int n)
    {
        List<int[]> faces = [[0, 2, 1], [0, 3, 2]];
        for (int i = 1; i < n; i++)
        {
            int l = 4 * i;
            faces.Add([l, l - 4, l + 1]);
            faces.Add([l + 1, l - 3, l + 2]);
            faces.Add([l + 2, l - 2, l + 3]);
            faces.Add([l + 3, l - 1, l]);
            faces.Add([l - 3, l + 1, l - 4]);
            faces.Add([l - 4, l, l - 1]);
            faces.Add([l - 1, l + 3, l - 2]);
            faces.Add([l - 2, l + 2, l - 3]);
            faces.Add([l, l + 1, l + 2]);
            faces.Add([l, l + 2, l + 3]);
        }
        return faces;
    }

    private static void ValidateCards(IReadOnlyList<HairCard> cards, bool cardColors)
    {
        foreach (HairCard card in cards)
        {
            ArgumentNullException.ThrowIfNull(card.Sequence);
            if (card.Sequence.Length % RowStride != 0)
                throw new ArgumentException("A card sequence must hold 8 values per control point.", nameof(cards));
            if (card.ShapeClass is not (0 or 1))
                throw new ArgumentException("A card's shape class must be 0 (square) or 1 (triangle).", nameof(cards));
            if ((card.Colors is not null) != cardColors)
                throw new ArgumentException("Card colours must be given on every card or on none.", nameof(cards));
            if (card.Colors is { } colors && (colors.Length == 0 || colors.Length % 3 != 0))
                throw new ArgumentException("A card's colours must be RGB triples.", nameof(cards));
        }
    }

    private static Vector3d[] Tangents(Vector3d[] points)
    {
        int n = points.Length;
        Vector3d[] tangents = new Vector3d[n];
        if (n < 5)
        {
            Vector3d chord = Vector3d.Unit(points[n - 1] - points[0]);
            Array.Fill(tangents, chord);
            return tangents;
        }
        for (int i = 0; i < n; i++)
        {
            int lo = Math.Min(Math.Max(i - 2, 0), n - 5);
            Vector3d sum = new(0, 0, 0);
            for (int k = 0; k < Stencil.Length; k++)
                sum += Stencil[k] * points[lo + k];
            tangents[i] = Vector3d.Unit(sum);
        }
        return tangents;
    }

    /// <summary>The reference colours of each vertex: the bins are interpolated in sRGB, as the reference does.</summary>
    private static double[] SrgbAlongCard(float[] colorsLinear, double[] vCoordinates)
    {
        int bins = colorsLinear.Length / 3;
        double[] binsSrgb = new double[colorsLinear.Length];
        for (int i = 0; i < binsSrgb.Length; i++)
            binsSrgb[i] = LinearToSrgb(colorsLinear[i]);

        double[] result = new double[3 * vCoordinates.Length];
        for (int v = 0; v < vCoordinates.Length; v++)
            for (int c = 0; c < 3; c++)
                result[3 * v + c] = InterpolateBins(vCoordinates[v], bins, binsSrgb, c);
        return result;
    }

    /// <summary>numpy.interp of one channel over bin centres (i + 0.5) / bins, constant beyond the first and last centre.</summary>
    private static double InterpolateBins(double x, int bins, double[] values, int channel)
    {
        double first = 0.5 / bins;
        double last = (bins - 0.5) / bins;
        if (x <= first)
            return values[channel];
        if (x >= last)
            return values[3 * (bins - 1) + channel];
        int j = 0;
        while (j < bins - 2 && (j + 1.5) / bins <= x)
            j++;
        double x0 = (j + 0.5) / bins;
        double x1 = (j + 1.5) / bins;
        double y0 = values[3 * j + channel];
        double y1 = values[3 * (j + 1) + channel];
        double slope = (y1 - y0) / (x1 - x0);
        return slope * (x - x0) + y0;
    }

    private static Vector3d At(double[] vertices, int index) =>
        new(vertices[3 * index], vertices[3 * index + 1], vertices[3 * index + 2]);

    private static double SrgbToLinear(double c) =>
        c <= SrgbThreshold ? c / 12.92 : Math.Pow((c + 0.055) / 1.055, 2.4);

    private static double LinearToSrgb(double c) =>
        c <= LinearSrgbThreshold ? 12.92 * c : 1.055 * Math.Pow(c, 1.0 / 2.4) - 0.055;
}
