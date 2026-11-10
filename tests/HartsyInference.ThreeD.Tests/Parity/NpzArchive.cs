using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Reads the arrays of a <c>.npz</c> file (a zip of <c>.npy</c> entries). Test support only; production code never reads it.</summary>
internal sealed class NpzArchive
{
    private static readonly byte[] NpyMagic = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];
    private static readonly Regex DescrPattern = new("'descr':\\s*'([^']+)'", RegexOptions.CultureInvariant);
    private static readonly Regex FortranPattern = new("'fortran_order':\\s*(True|False)", RegexOptions.CultureInvariant);
    private static readonly Regex ShapePattern = new("'shape':\\s*\\(([^)]*)\\)", RegexOptions.CultureInvariant);

    private readonly Dictionary<string, NpzArray> _arrays;

    private NpzArchive(Dictionary<string, NpzArray> arrays)
    {
        _arrays = arrays;
    }

    /// <summary>Reads every <c>.npy</c> entry into memory.</summary>
    internal static NpzArchive Open(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Parity fixture not found: {path}.", path);

        Dictionary<string, NpzArray> arrays = new(StringComparer.Ordinal);
        using ZipArchive zip = ZipFile.OpenRead(path);
        foreach (ZipArchiveEntry entry in zip.Entries)
        {
            if (!entry.FullName.EndsWith(".npy", StringComparison.Ordinal))
                continue;
            using MemoryStream bytes = new();
            using (Stream stream = entry.Open())
                stream.CopyTo(bytes);
            arrays.Add(entry.FullName[..^4], ParseNpy(bytes.ToArray(), entry.FullName));
        }

        return new NpzArchive(arrays);
    }

    /// <summary>The named array; throws when the fixture does not hold it.</summary>
    internal NpzArray Get(string name)
    {
        if (!_arrays.TryGetValue(name, out NpzArray? array))
            throw new InvalidDataException($"Parity fixture has no array '{name}'; it holds {string.Join(", ", _arrays.Keys)}.");
        return array;
    }

    private static NpzArray ParseNpy(byte[] npy, string entryName)
    {
        if (npy.Length < 10 || !npy.AsSpan(0, NpyMagic.Length).SequenceEqual(NpyMagic))
            throw new InvalidDataException($"{entryName} is not a .npy entry.");
        byte major = npy[6];
        if (major is < 1 or > 3)
            throw new InvalidDataException($"{entryName} has unsupported .npy version {major}.");

        long headerLength = major == 1
            ? BinaryPrimitives.ReadUInt16LittleEndian(npy.AsSpan(8))
            : BinaryPrimitives.ReadUInt32LittleEndian(npy.AsSpan(8));
        int headerStart = major == 1 ? 10 : 12;
        long dataStart = headerStart + headerLength;
        if (dataStart > npy.Length)
            throw new InvalidDataException($"{entryName} header runs past the end of the entry.");

        string header = Encoding.Latin1.GetString(npy, headerStart, (int)headerLength);
        string descr = Field(DescrPattern, header, entryName, "descr");
        if (Field(FortranPattern, header, entryName, "fortran_order") != "False")
            throw new InvalidDataException($"{entryName} is Fortran-ordered; only C-ordered arrays are read.");
        long[] shape = ParseShape(Field(ShapePattern, header, entryName, "shape"), entryName);

        long elements = 1;
        foreach (long dim in shape)
            elements *= dim;
        int itemSize = ItemSize(descr, entryName);
        byte[] data = npy[(int)dataStart..];
        if (data.LongLength != elements * itemSize)
            throw new InvalidDataException($"{entryName} holds {data.LongLength} data bytes; expected {elements * itemSize}.");
        return new NpzArray(descr, shape, data);
    }

    private static string Field(Regex pattern, string header, string entryName, string field)
    {
        Match match = pattern.Match(header);
        if (!match.Success)
            throw new InvalidDataException($"{entryName} header has no '{field}' field.");
        return match.Groups[1].Value;
    }

    private static long[] ParseShape(string text, string entryName)
    {
        string[] parts = text.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        long[] shape = new long[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            if (!long.TryParse(parts[i], NumberStyles.Integer, CultureInfo.InvariantCulture, out shape[i]))
                throw new InvalidDataException($"{entryName} has a non-integer shape entry '{parts[i]}'.");
        }

        return shape;
    }

    private static int ItemSize(string descr, string entryName) => descr switch
    {
        "<f4" or "<i4" => 4,
        "<f8" => 8,
        "|b1" or "|u1" => 1,
        _ => throw new InvalidDataException($"{entryName} has unsupported dtype '{descr}'."),
    };
}
