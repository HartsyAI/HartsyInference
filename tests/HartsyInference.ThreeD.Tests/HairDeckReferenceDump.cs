using System.Runtime.InteropServices;
using System.Text.Json;

namespace HartsyInference.ThreeD.Tests;

/// <summary>Reader of the raw blobs of <c>tools/parity/dump_hairdeck_core.py</c>: <c>manifest.json</c> names each tensor's dtype,
/// shape and file; the files are little-endian and row-major.</summary>
public sealed class HairDeckReferenceDump
{
    private readonly string _dir;
    private readonly Dictionary<string, (string DType, int[] Shape, string File)> _entries = [];

    private HairDeckReferenceDump(string dir)
    {
        _dir = dir;
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "manifest.json")));
        foreach (JsonProperty p in doc.RootElement.GetProperty("tensors").EnumerateObject())
        {
            int[] shape = p.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            _entries[p.Name] = (p.Value.GetProperty("dtype").GetString()!, shape, p.Value.GetProperty("file").GetString()!);
        }
    }

    public static HairDeckReferenceDump Open(string dir) => new(dir);

    public int[] Shape(string name) => Entry(name).Shape;

    /// <summary>The tensor as float32 values (float64 dumps are narrowed).</summary>
    public float[] Floats(string name)
    {
        (string dtype, _, string file) = Entry(name);
        byte[] bytes = File.ReadAllBytes(Path.Combine(_dir, file));
        return dtype switch
        {
            "float32" => MemoryMarshal.Cast<byte, float>(bytes).ToArray(),
            "float64" => MemoryMarshal.Cast<byte, double>(bytes).ToArray().Select(v => (float)v).ToArray(),
            "uint8" => bytes.Select(b => (float)b).ToArray(),
            "int64" => MemoryMarshal.Cast<byte, long>(bytes).ToArray().Select(v => (float)v).ToArray(),
            _ => throw new InvalidDataException($"Unsupported dump dtype '{dtype}' for '{name}'."),
        };
    }

    /// <summary>The tensor as float64 values (float32 dumps are widened).</summary>
    public double[] Doubles(string name)
    {
        (string dtype, _, string file) = Entry(name);
        byte[] bytes = File.ReadAllBytes(Path.Combine(_dir, file));
        return dtype switch
        {
            "float32" => MemoryMarshal.Cast<byte, float>(bytes).ToArray().Select(v => (double)v).ToArray(),
            "float64" => MemoryMarshal.Cast<byte, double>(bytes).ToArray(),
            "int64" => MemoryMarshal.Cast<byte, long>(bytes).ToArray().Select(v => (double)v).ToArray(),
            "uint8" => bytes.Select(b => (double)b).ToArray(),
            _ => throw new InvalidDataException($"Unsupported dump dtype '{dtype}' for '{name}'."),
        };
    }

    /// <summary>The tensor as int64 values.</summary>
    public long[] Longs(string name)
    {
        (string dtype, _, string file) = Entry(name);
        if (dtype != "int64") throw new InvalidDataException($"'{name}' is {dtype}, not int64.");
        return MemoryMarshal.Cast<byte, long>(File.ReadAllBytes(Path.Combine(_dir, file))).ToArray();
    }

    private (string DType, int[] Shape, string File) Entry(string name) =>
        _entries.TryGetValue(name, out (string DType, int[] Shape, string File) e)
            ? e
            : throw new KeyNotFoundException($"HairDeck reference dump has no tensor '{name}'.");
}

/// <summary>Element-wise error of an engine result against its reference: max abs, RMS, and max abs over the peak |reference|.</summary>
public readonly record struct HairDeckError(double MaxAbs, double Rms, double Peak)
{
    public double PeakRelative => Peak > 0 ? MaxAbs / Peak : MaxAbs;

    public override string ToString() => $"max abs {MaxAbs:G6}, rms {Rms:G6}, peak {Peak:G6}, peak-relative {PeakRelative:G6}";

    public static HairDeckError Compare(ReadOnlySpan<float> ours, ReadOnlySpan<double> reference)
    {
        if (ours.Length != reference.Length)
            throw new InvalidOperationException($"Compared {ours.Length} values against {reference.Length} reference values.");
        double maxAbs = 0, sumSq = 0, peak = 0;
        for (int i = 0; i < ours.Length; i++)
        {
            double d = Math.Abs(ours[i] - reference[i]);
            maxAbs = Math.Max(maxAbs, d);
            sumSq += d * d;
            peak = Math.Max(peak, Math.Abs(reference[i]));
        }
        return new HairDeckError(maxAbs, Math.Sqrt(sumSq / Math.Max(1, ours.Length)), peak);
    }
}
