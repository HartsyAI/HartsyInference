using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>Reads the vertices of a PLY file as a point cloud: positions, and colours from <c>red/green/blue</c> or, for a Gaussian-splat file, from the DC spherical-harmonic terms (<c>f_dc_0..2</c>). A splat's faint points (opacity below 5 %) are dropped. Binary little-endian and ASCII files are read; faces and every property that is not a position, colour or opacity are skipped.</summary>
public static class PointPlyReader
{
    private const float ShC0 = 0.28209479177387814f;

    /// <summary>The points, whether the file is a splat cloud, and how many faint splats were dropped.</summary>
    public sealed record Result(PointCloud Cloud, bool IsSplat, int Dropped);

    /// <summary>Parses <paramref name="ply"/>.</summary>
    /// <exception cref="HartsyInferenceException">The file is not a PLY, is big-endian, has no x, y and z properties, or is shorter than its header says.</exception>
    public static Result Read(byte[] ply)
    {
        ArgumentNullException.ThrowIfNull(ply);
        int headerEnd = IndexOfHeaderEnd(ply, out int bodyStart);
        if (headerEnd < 0 || !Encoding.ASCII.GetString(ply, 0, Math.Min(3, ply.Length)).Equals("ply", StringComparison.Ordinal))
            throw new HartsyInferenceException("The file is not a PLY file (no header).");
        string[] lines = Encoding.ASCII.GetString(ply, 0, headerEnd).Split('\n');
        string format = "";
        int count = 0;
        bool inVertex = false, vertexSeen = false;
        List<(string Type, string Name)> properties = [];
        foreach (string raw in lines)
        {
            string[] parts = raw.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts is ["format", _, ..])
            {
                format = parts[1];
            }
            else if (parts is ["element", _, _])
            {
                // The body is read from the end of the header, so the vertex element has to be the first one.
                if (!vertexSeen && parts[1] != "vertex")
                    throw new HartsyInferenceException($"The PLY element '{parts[1]}' comes before the vertex element; only files whose first element is vertex are read.");
                inVertex = parts[1] == "vertex" && !vertexSeen;
                if (inVertex)
                {
                    vertexSeen = true;
                    count = int.TryParse(parts[2], NumberStyles.None, CultureInfo.InvariantCulture, out int n) ? n : throw new HartsyInferenceException("The PLY vertex count is not a number.");
                }
            }
            else if (inVertex && parts is ["property", ..])
            {
                if (parts.Length == 5 && parts[1] == "list")
                    throw new HartsyInferenceException("A PLY vertex with a list property is not supported.");
                properties.Add((parts[1], parts[^1]));
            }
        }
        if (format is not ("binary_little_endian" or "ascii"))
            throw new HartsyInferenceException($"PLY format '{format}' is not read; expected binary_little_endian or ascii.");
        int x = properties.FindIndex(p => p.Name == "x"), y = properties.FindIndex(p => p.Name == "y"), z = properties.FindIndex(p => p.Name == "z");
        if (x < 0 || y < 0 || z < 0 || count == 0)
            throw new HartsyInferenceException("The PLY file has no vertices with x, y and z properties.");
        // Every vertex takes at least one byte (a binary one) or one line, so a count beyond the file's length is a lie, and nothing is sized from it.
        if (count > ply.Length)
            throw new HartsyInferenceException($"The PLY header declares {count} vertices but the file holds only {ply.Length} bytes.");

        int[] rgb = [.. new[] { "red", "green", "blue" }.Select(n => properties.FindIndex(p => p.Name == n))];
        int[] dc = [.. new[] { "f_dc_0", "f_dc_1", "f_dc_2" }.Select(n => properties.FindIndex(p => p.Name == n))];
        int opacity = properties.FindIndex(p => p.Name == "opacity");
        bool splat = dc.All(i => i >= 0);
        bool coloured = splat || rgb.All(i => i >= 0);
        // Integer colours are scaled to [0, 1] by their type's range; float ones are taken as they are.
        double colourScale = splat || !coloured ? 1.0 : properties[rgb[0]].Type switch
        {
            "uchar" or "uint8" or "char" or "int8" => 255.0,
            "ushort" or "uint16" or "short" or "int16" => 65535.0,
            _ => 1.0,
        };

        // Only the properties that are drawn are kept: a splat has dozens per point and there can be millions of points.
        List<int> wanted = [x, y, z];
        if (splat)
            wanted.AddRange(dc);
        else if (coloured)
            wanted.AddRange(rgb);
        if (splat && opacity >= 0)
            wanted.Add(opacity);
        double[] values = format == "ascii" ? ReadAscii(ply, bodyStart, count, properties.Count, wanted) : ReadBinary(ply, bodyStart, count, properties, wanted);
        int columns = wanted.Count;

        float[] positions = new float[count * 3];
        float[]? colours = coloured ? new float[count * 3] : null;
        int kept = 0, dropped = 0;
        for (int i = 0; i < count; i++)
        {
            int row = i * columns;
            if (splat && opacity >= 0 && 1.0 / (1.0 + Math.Exp(-values[row + columns - 1])) < 0.05)
            {
                dropped++;
                continue;
            }
            for (int c = 0; c < 3; c++)
                positions[kept * 3 + c] = (float)values[row + c];
            if (colours is not null)
            {
                for (int c = 0; c < 3; c++)
                {
                    double v = values[row + 3 + c];
                    colours[kept * 3 + c] = (float)Math.Clamp(splat ? 0.5 + ShC0 * v : v / colourScale, 0.0, 1.0);
                }
            }
            kept++;
        }
        if (kept == 0)
            throw new HartsyInferenceException("Every splat in the file is fainter than 5 % opacity; there is nothing to draw.");
        Array.Resize(ref positions, kept * 3);
        if (colours is not null)
            Array.Resize(ref colours, kept * 3);
        return new Result(new PointCloud { Positions = positions, Colors = colours }, splat, dropped);
    }

    /// <summary>The <paramref name="wanted"/> properties of every vertex, row by row.</summary>
    private static double[] ReadBinary(byte[] ply, int start, int count, List<(string Type, string Name)> properties, List<int> wanted)
    {
        int[] sizes = [.. properties.Select(p => SizeOf(p.Type))];
        int[] offsets = new int[sizes.Length];
        for (int p = 1; p < sizes.Length; p++)
            offsets[p] = offsets[p - 1] + sizes[p - 1];
        long stride = sizes.Sum();
        if (ply.LongLength < start + stride * count)
            throw new HartsyInferenceException($"The PLY file is shorter than its header says ({count} vertices of {stride} bytes).");
        double[] values = new double[(long)count * wanted.Count];
        for (int i = 0; i < count; i++)
        {
            long row = start + i * stride;
            for (int w = 0; w < wanted.Count; w++)
            {
                double value = Decode(properties[wanted[w]].Type, ply.AsSpan((int)(row + offsets[wanted[w]])));
                // A NaN or infinite coordinate would make the bounds, and so the camera, meaningless; the ASCII path refuses them too.
                if (!double.IsFinite(value))
                    throw new HartsyInferenceException($"PLY vertex {i + 1} holds a value ({properties[wanted[w]].Name}) that is not a finite number.");
                values[(long)i * wanted.Count + w] = value;
            }
        }
        return values;
    }

    private static double[] ReadAscii(byte[] ply, int start, int count, int properties, List<int> wanted)
    {
        string[] lines = Encoding.ASCII.GetString(ply, start, ply.Length - start).Split('\n');
        if (lines.Length < count)
            throw new HartsyInferenceException("The PLY file has fewer vertex lines than its header says.");
        double[] values = new double[(long)count * wanted.Count];
        for (int i = 0; i < count; i++)
        {
            string[] tokens = lines[i].Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length < properties)
                throw new HartsyInferenceException($"PLY vertex line {i + 1} has fewer than {properties} values.");
            for (int w = 0; w < wanted.Count; w++)
            {
                if (!double.TryParse(tokens[wanted[w]], NumberStyles.Float, CultureInfo.InvariantCulture, out double d) || !double.IsFinite(d))
                    throw new HartsyInferenceException($"PLY vertex line {i + 1} holds a value that is not a finite number.");
                values[(long)i * wanted.Count + w] = d;
            }
        }
        return values;
    }

    private static int SizeOf(string type) => type switch
    {
        "char" or "uchar" or "int8" or "uint8" => 1,
        "short" or "ushort" or "int16" or "uint16" => 2,
        "int" or "uint" or "float" or "int32" or "uint32" or "float32" => 4,
        "double" or "float64" => 8,
        _ => throw new HartsyInferenceException($"PLY property type '{type}' is not supported."),
    };

    private static double Decode(string type, ReadOnlySpan<byte> data) => type switch
    {
        "char" or "int8" => (sbyte)data[0],
        "uchar" or "uint8" => data[0],
        "short" or "int16" => BinaryPrimitives.ReadInt16LittleEndian(data),
        "ushort" or "uint16" => BinaryPrimitives.ReadUInt16LittleEndian(data),
        "int" or "int32" => BinaryPrimitives.ReadInt32LittleEndian(data),
        "uint" or "uint32" => BinaryPrimitives.ReadUInt32LittleEndian(data),
        "float" or "float32" => BinaryPrimitives.ReadSingleLittleEndian(data),
        _ => BinaryPrimitives.ReadDoubleLittleEndian(data),
    };

    /// <summary>The length of the header text and where the body starts, or -1 when there is no <c>end_header</c> line in the first 64 KiB.</summary>
    private static int IndexOfHeaderEnd(byte[] ply, out int bodyStart)
    {
        byte[] marker = Encoding.ASCII.GetBytes("end_header");
        int limit = Math.Min(ply.Length, 1 << 16) - marker.Length;
        for (int i = 0; i <= limit; i++)
        {
            if (!ply.AsSpan(i, marker.Length).SequenceEqual(marker))
                continue;
            int end = i + marker.Length;
            if (end < ply.Length && ply[end] == '\r')
                end++;
            bodyStart = end < ply.Length && ply[end] == '\n' ? end + 1 : -1;
            return bodyStart < 0 ? -1 : i;
        }
        bodyStart = -1;
        return -1;
    }
}
