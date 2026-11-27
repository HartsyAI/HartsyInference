namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>One array of an <see cref="NpzArchive"/>: its numpy dtype string, C-ordered shape and raw little-endian bytes.</summary>
internal sealed record NpzArray(string Descr, long[] Shape, byte[] Data)
{
    /// <summary>Checks the shape and returns this array, so a fixture read can be chained.</summary>
    internal NpzArray RequireShape(params long[] expected)
    {
        if (expected.Length != Shape.Length || !expected.AsSpan().SequenceEqual(Shape))
            throw new InvalidDataException($"Fixture array has shape [{string.Join(", ", Shape)}]; expected [{string.Join(", ", expected)}].");
        return this;
    }

    internal float[] ToFloat32()
    {
        RequireLittleEndianHost();
        if (Descr != "<f4")
            throw new InvalidDataException($"Fixture array is {Descr}; expected <f4.");
        float[] values = new float[Data.Length / sizeof(float)];
        Buffer.BlockCopy(Data, 0, values, 0, Data.Length);
        return values;
    }

    internal int[] ToInt32()
    {
        RequireLittleEndianHost();
        if (Descr != "<i4")
            throw new InvalidDataException($"Fixture array is {Descr}; expected <i4.");
        int[] values = new int[Data.Length / sizeof(int)];
        Buffer.BlockCopy(Data, 0, values, 0, Data.Length);
        return values;
    }

    /// <summary>Boolean or uint8 flags; any non-zero byte is true.</summary>
    internal bool[] ToBool()
    {
        if (Descr != "|b1" && Descr != "|u1")
            throw new InvalidDataException($"Fixture array is {Descr}; expected a boolean or uint8 array.");
        bool[] values = new bool[Data.Length];
        for (int i = 0; i < values.Length; i++)
            values[i] = Data[i] != 0;
        return values;
    }

    private static void RequireLittleEndianHost()
    {
        if (!BitConverter.IsLittleEndian)
            throw new NotSupportedException("Fixture reader assumes a little-endian host.");
    }
}
