using System.Buffers.Binary;
using System.Globalization;
using System.Text;

namespace HartsyInference.ThreeD.Io;

/// <summary>
/// Reads the point clouds HairDeck conditions on: binary little-endian PLY with one <c>vertex</c> element whose first six
/// properties are <c>x y z nx ny nz</c> (any scalar type; the rest is ignored, colours included). Matches the layout of
/// upstream <c>hairdeck/ply.py</c> and CHARM's <c>sample.py</c> output (double x y z nx ny nz, uchar red green blue).
/// </summary>
public static class PointCloudPlyReader
{
    /// <summary>A point cloud: positions and normals, <c>[N, 6]</c> row-major (x y z nx ny nz), in the file's units.</summary>
    public sealed record PointCloud(double[] Data, int Count);

    private sealed record Property(string Name, string Type, int Size);

    /// <summary>Parses a PLY file image. Throws <see cref="InvalidDataException"/> for any layout HairDeck does not accept.</summary>
    public static PointCloud Read(ReadOnlySpan<byte> file)
    {
        int headerEnd = IndexOf(file, "end_header\n"u8);
        if (headerEnd < 0)
        {
            throw new InvalidDataException("Not a PLY file: no end_header line.");
        }

        int bodyStart = headerEnd + "end_header\n"u8.Length;
        string header = Encoding.ASCII.GetString(file[..bodyStart]);
        string[] lines = header.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (lines.Length < 2 || lines[0] != "ply")
        {
            throw new InvalidDataException("Not a PLY file: missing the ply magic line.");
        }
        if (!lines.Any(l => l == "format binary_little_endian 1.0"))
        {
            throw new InvalidDataException("PLY must be binary little-endian (format binary_little_endian 1.0).");
        }

        int vertexCount = -1;
        List<Property> props = new();
        bool inVertex = false;
        bool sawVertexElement = false;
        foreach (string line in lines)
        {
            string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0)
            {
                continue;
            }
            switch (parts[0])
            {
                case "element":
                    inVertex = parts.Length == 3 && parts[1] == "vertex";
                    if (inVertex)
                    {
                        if (sawVertexElement)
                        {
                            throw new InvalidDataException("PLY has more than one vertex element.");
                        }
                        sawVertexElement = true;
                        vertexCount = int.Parse(parts[2], CultureInfo.InvariantCulture);
                    }
                    else if (sawVertexElement)
                    {
                        // Elements after the vertex element are not read; the vertex data comes first in the body.
                        inVertex = false;
                    }
                    break;
                case "property" when inVertex:
                    if (parts.Length != 3 || parts[1] == "list")
                    {
                        throw new InvalidDataException("Vertex properties must be scalars.");
                    }
                    props.Add(new Property(parts[2], parts[1], SizeOf(parts[1])));
                    break;
            }
        }

        if (!sawVertexElement || vertexCount < 0)
        {
            throw new InvalidDataException("PLY has no vertex element.");
        }
        string[] required = ["x", "y", "z", "nx", "ny", "nz"];
        if (props.Count < 6 || !props.Take(6).Select(p => p.Name).SequenceEqual(required))
        {
            throw new InvalidDataException("The first six vertex properties must be x y z nx ny nz.");
        }

        int stride = props.Sum(p => p.Size);
        long needed = (long)vertexCount * stride;
        ReadOnlySpan<byte> body = file[bodyStart..];
        if (body.Length < needed)
        {
            throw new InvalidDataException($"PLY body is {body.Length} bytes; {vertexCount} vertices need {needed}.");
        }

        double[] data = new double[(long)vertexCount * 6];
        int[] offsets = new int[6];
        int off = 0;
        for (int i = 0; i < 6; i++)
        {
            offsets[i] = off;
            off += props[i].Size;
        }

        for (int v = 0; v < vertexCount; v++)
        {
            ReadOnlySpan<byte> row = body.Slice(v * stride, stride);
            for (int c = 0; c < 6; c++)
            {
                data[v * 6 + c] = ReadScalar(row[offsets[c]..], props[c].Type);
            }
        }

        return new PointCloud(data, vertexCount);
    }

    /// <summary>Reads a PLY file from disk.</summary>
    public static PointCloud ReadFile(string path) => Read(File.ReadAllBytes(path));

    private static int SizeOf(string type) => type switch
    {
        "char" or "int8" or "uchar" or "uint8" => 1,
        "short" or "int16" or "ushort" or "uint16" => 2,
        "int" or "int32" or "uint" or "uint32" or "float" or "float32" => 4,
        "double" or "float64" => 8,
        _ => throw new InvalidDataException($"Unknown PLY property type '{type}'."),
    };

    private static double ReadScalar(ReadOnlySpan<byte> s, string type) => type switch
    {
        "char" or "int8" => (sbyte)s[0],
        "uchar" or "uint8" => s[0],
        "short" or "int16" => BinaryPrimitives.ReadInt16LittleEndian(s),
        "ushort" or "uint16" => BinaryPrimitives.ReadUInt16LittleEndian(s),
        "int" or "int32" => BinaryPrimitives.ReadInt32LittleEndian(s),
        "uint" or "uint32" => BinaryPrimitives.ReadUInt32LittleEndian(s),
        "float" or "float32" => BinaryPrimitives.ReadSingleLittleEndian(s),
        "double" or "float64" => BinaryPrimitives.ReadDoubleLittleEndian(s),
        _ => throw new InvalidDataException($"Unknown PLY property type '{type}'."),
    };

    private static int IndexOf(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle)
    {
        int i = haystack.IndexOf(needle);
        return i;
    }
}
