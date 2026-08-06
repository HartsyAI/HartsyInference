using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Shared IO/compare helpers for the annotator parity tests (HED, lineart).</summary>
internal static unsafe class AnnotatorParityIo
{
    public static float[] ReadF32(string path, long expectedCount)
    {
        byte[] raw = File.ReadAllBytes(path);
        Assert.Equal(expectedCount * 4, raw.Length);
        float[] data = new float[expectedCount];
        Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
        return data;
    }

    public static void WriteF32(string path, Tensor t)
    {
        ReadOnlySpan<byte> bytes = new((void*)t.DataPointer, (int)(t.ElementCount * 4));
        File.WriteAllBytes(path, bytes.ToArray());
    }

    public static (double AvgErr, double MaxErr, double RefMeanAbs) Compare(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double sumErr = 0, maxErr = 0, sumRefAbs = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double d = Math.Abs(actual[i] - expected[i]);
            sumErr += d;
            if (d > maxErr) maxErr = d;
            sumRefAbs += Math.Abs(expected[i]);
        }
        return (sumErr / actual.Length, maxErr, sumRefAbs / actual.Length);
    }

    /// <summary>Fraction of positions whose bytes differ — for quantized u8 outputs where float noise at a
    /// quantization boundary legitimately flips single codes.</summary>
    public static double MismatchFraction(ReadOnlySpan<byte> actual, ReadOnlySpan<byte> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        long mismatched = 0;
        for (int i = 0; i < actual.Length; i++)
            if (actual[i] != expected[i]) mismatched++;
        return (double)mismatched / actual.Length;
    }

    /// <summary>Unit-map floats → the u8 codes they sit on (the maps are quantized to the u8 grid).</summary>
    public static byte[] UnitToBytes(ReadOnlySpan<float> unit)
    {
        byte[] bytes = new byte[unit.Length];
        for (int i = 0; i < unit.Length; i++)
            bytes[i] = (byte)Math.Clamp((int)MathF.Round(unit[i] * 255f), 0, 255);
        return bytes;
    }
}
