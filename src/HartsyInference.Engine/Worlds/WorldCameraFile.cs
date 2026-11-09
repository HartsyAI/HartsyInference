using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using System.Text.Json;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Worlds;

/// <summary>Reads a camera path from a file (<c>.json</c> or <c>.npy</c>) and parses the <c>fx,fy,cx,cy</c> intrinsics text.</summary>
/// <remarks>
/// JSON: <c>{"poses": [...], "intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...}}</c>, or a bare array of poses.
/// Each pose is 16 row-major numbers, or four rows of four. NumPy: a little-endian float32 or float64 array shaped
/// <c>(N, 16)</c> or <c>(N, 4, 4)</c>, C order. Both carry camera-to-world matrices in OpenCV convention (+X right,
/// +Y down, +Z forward), one per frame; intrinsics are pixels of the original seed image.
/// </remarks>
public static class WorldCameraFile
{
    /// <summary>Reads the camera path at <paramref name="path"/>, choosing the format from its extension.</summary>
    /// <exception cref="FileNotFoundException">The file does not exist.</exception>
    /// <exception cref="InvalidDataException">The file is not a well-formed camera path.</exception>
    public static WorldCameraPath Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Camera path file not found: {path}", path);
        }
        return Parse(File.ReadAllBytes(path), Path.GetExtension(path));
    }

    /// <summary>Parses camera-path bytes whose format is named by <paramref name="extension"/> (<c>.json</c> or <c>.npy</c>).</summary>
    /// <exception cref="InvalidDataException">The bytes are not a well-formed camera path in that format.</exception>
    public static WorldCameraPath Parse(byte[] bytes, string extension) => extension.ToLowerInvariant() switch
    {
        ".npy" => ParseNpy(bytes),
        ".json" => ParseJson(bytes),
        _ => throw new InvalidDataException($"A camera path must be a .npy or .json file; got '{extension}'."),
    };

    /// <summary>Parses intrinsics written as <c>fx,fy,cx,cy</c>, in pixels of the original seed image.</summary>
    /// <exception cref="ArgumentException">The text is not four comma-separated numbers.</exception>
    public static WorldIntrinsics ParseIntrinsics(string text)
    {
        string[] parts = (text ?? "").Split(',', StringSplitOptions.TrimEntries);
        if (parts.Length != 4
            || !double.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double fx)
            || !double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double fy)
            || !double.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out double cx)
            || !double.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out double cy))
        {
            throw new ArgumentException($"Intrinsics must be fx,fy,cx,cy in pixels of the seed image; got '{text}'.", nameof(text));
        }
        return new WorldIntrinsics { Fx = fx, Fy = fy, Cx = cx, Cy = cy };
    }

    private static WorldCameraPath ParseJson(byte[] bytes)
    {
        using JsonDocument document = JsonDocument.Parse(bytes);
        JsonElement root = document.RootElement;
        JsonElement posesElement;
        WorldIntrinsics? intrinsics = null;
        if (root.ValueKind == JsonValueKind.Array)
        {
            posesElement = root;
        }
        else if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("poses", out JsonElement poses))
        {
            posesElement = poses;
            if (root.TryGetProperty("intrinsics", out JsonElement intrinsicsElement) && intrinsicsElement.ValueKind != JsonValueKind.Null)
            {
                intrinsics = ReadIntrinsics(intrinsicsElement);
            }
        }
        else
        {
            throw new InvalidDataException("A camera path JSON file is an array of poses or an object with a \"poses\" array.");
        }
        List<float[]> list = [];
        foreach (JsonElement pose in posesElement.EnumerateArray())
        {
            list.Add(ReadPose(pose));
        }
        return new WorldCameraPath { Poses = list, Intrinsics = intrinsics };
    }

    private static float[] ReadPose(JsonElement pose)
    {
        if (pose.ValueKind != JsonValueKind.Array)
        {
            throw new InvalidDataException("Each camera pose must be an array of numbers.");
        }
        List<float> values = [];
        foreach (JsonElement item in pose.EnumerateArray())
        {
            if (item.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement number in item.EnumerateArray())
                {
                    values.Add(ReadNumber(number));
                }
            }
            else
            {
                values.Add(ReadNumber(item));
            }
        }
        return values.ToArray();
    }

    private static float ReadNumber(JsonElement number)
    {
        if (number.ValueKind != JsonValueKind.Number)
        {
            throw new InvalidDataException("Camera pose values must be numbers.");
        }
        return (float)number.GetDouble();
    }

    private static WorldIntrinsics ReadIntrinsics(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Object
            || !element.TryGetProperty("fx", out JsonElement fx) || !element.TryGetProperty("fy", out JsonElement fy)
            || !element.TryGetProperty("cx", out JsonElement cx) || !element.TryGetProperty("cy", out JsonElement cy))
        {
            throw new InvalidDataException("Intrinsics must be an object with fx, fy, cx and cy.");
        }
        return new WorldIntrinsics { Fx = fx.GetDouble(), Fy = fy.GetDouble(), Cx = cx.GetDouble(), Cy = cy.GetDouble() };
    }

    private static WorldCameraPath ParseNpy(byte[] bytes)
    {
        if (bytes.Length < 10 || bytes[0] != 0x93 || !bytes.AsSpan(1, 5).SequenceEqual("NUMPY"u8))
        {
            throw new InvalidDataException("The camera path .npy file has no NUMPY header.");
        }
        int major = bytes[6];
        int headerLength;
        int headerStart;
        if (major == 1)
        {
            headerLength = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(8));
            headerStart = 10;
        }
        else if (major is 2 or 3)
        {
            headerLength = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(8)));
            headerStart = 12;
        }
        else
        {
            throw new InvalidDataException($"Unsupported .npy format version {major}.");
        }
        if (headerStart + headerLength > bytes.Length)
        {
            throw new InvalidDataException("The camera path .npy header is truncated.");
        }
        string header = Encoding.UTF8.GetString(bytes, headerStart, headerLength);
        if (header.Contains("'fortran_order': True", StringComparison.Ordinal))
        {
            throw new InvalidDataException("Fortran-order camera path arrays are not supported; save the array in C order.");
        }
        string descr = HeaderField(header, "'descr': '", '\'');
        int elementSize = descr switch
        {
            "<f4" => 4,
            "<f8" => 8,
            _ => throw new InvalidDataException($"Camera path .npy dtype '{descr}' is not little-endian float32 or float64."),
        };
        int[] shape = ParseShape(HeaderField(header, "'shape': (", ')'));
        int count = shape.Length switch
        {
            2 when shape[1] == 16 => shape[0],
            3 when shape[1] == 4 && shape[2] == 4 => shape[0],
            _ => throw new InvalidDataException("A camera path .npy array must be shaped (N, 16) or (N, 4, 4)."),
        };
        long dataStart = headerStart + headerLength;
        if (dataStart + (long)count * 16 * elementSize > bytes.Length)
        {
            throw new InvalidDataException("The camera path .npy data is shorter than its header declares.");
        }
        ReadOnlySpan<byte> data = bytes.AsSpan((int)dataStart);
        List<float[]> poses = new List<float[]>(count);
        for (int i = 0; i < count; i++)
        {
            float[] pose = new float[16];
            for (int k = 0; k < 16; k++)
            {
                int offset = (i * 16 + k) * elementSize;
                pose[k] = elementSize == 4
                    ? BinaryPrimitives.ReadSingleLittleEndian(data[offset..])
                    : (float)BinaryPrimitives.ReadDoubleLittleEndian(data[offset..]);
            }
            poses.Add(pose);
        }
        return new WorldCameraPath { Poses = poses };
    }

    private static string HeaderField(string header, string key, char terminator)
    {
        int start = header.IndexOf(key, StringComparison.Ordinal);
        if (start < 0)
        {
            throw new InvalidDataException($"The camera path .npy header has no {key.Trim('\'', ':', ' ', '(')} entry.");
        }
        start += key.Length;
        int end = header.IndexOf(terminator, start);
        if (end < 0)
        {
            throw new InvalidDataException("The camera path .npy header is malformed.");
        }
        return header[start..end];
    }

    private static int[] ParseShape(string text)
    {
        List<int> dims = [];
        foreach (string part in text.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
        {
            if (!int.TryParse(part, NumberStyles.None, CultureInfo.InvariantCulture, out int dim) || dim < 1)
            {
                throw new InvalidDataException($"The camera path .npy shape '{text}' is not a list of positive sizes.");
            }
            dims.Add(dim);
        }
        return dims.ToArray();
    }
}
