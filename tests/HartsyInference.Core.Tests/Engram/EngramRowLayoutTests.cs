using HartsyInference.Core.Engram;
using Xunit;

namespace HartsyInference.Core.Tests.Engram;

/// <summary>Row decode of the three storage layouts against python dequantization (fixtures dumped by dump_engram_rows_fixtures.py).</summary>
public sealed class EngramRowLayoutTests
{
    [Fact]
    public void Official_DecodesEdgeRowsLikePython()
    {
        System.Text.Json.JsonElement edge = EngramFixtures.RowDecode.GetProperty("edge");
        System.Text.Json.JsonElement official = edge.GetProperty("official");
        byte[] fp8 = EngramFixtures.Bytes(official.GetProperty("fp8"));
        byte[] e8m0 = EngramFixtures.Bytes(official.GetProperty("e8m0"));
        ushort[] expected = EngramFixtures.Bf16(official.GetProperty("bf16"));
        OfficialFp8E8M0RowLayout layout = new(1000, 664, 5000);
        int rows = edge.GetProperty("rows").GetInt32();

        for (int r = 0; r < rows; r++)
        {
            byte[] packed = [.. fp8.AsSpan(r * 256, 256), .. e8m0.AsSpan(r * 8, 8)];
            ushort[] actual = new ushort[256];
            layout.DecodeRow(packed, actual);
            EngramFixtures.AssertBf16Equal(expected.AsSpan(r * 256, 256), actual, $"official edge row {r}");
        }
    }

    [Fact]
    public void Gguf_DecodesEdgeRowsLikePython()
    {
        System.Text.Json.JsonElement edge = EngramFixtures.RowDecode.GetProperty("edge");
        System.Text.Json.JsonElement gguf = edge.GetProperty("gguf");
        byte[] raw = EngramFixtures.Bytes(gguf.GetProperty("row264"));
        ushort[] expected = EngramFixtures.Bf16(gguf.GetProperty("bf16"));
        GgufRow264Layout layout = new(1000, 16384);

        for (int r = 0; r < edge.GetProperty("rows").GetInt32(); r++)
        {
            ushort[] actual = new ushort[256];
            layout.DecodeRow(raw.AsSpan(r * 264, 264), actual);
            EngramFixtures.AssertBf16Equal(expected.AsSpan(r * 256, 256), actual, $"gguf edge row {r}");
        }
    }

    [Fact]
    public void Mlx_DecodesEdgeRowsLikePython()
    {
        System.Text.Json.JsonElement edge = EngramFixtures.RowDecode.GetProperty("edge");
        System.Text.Json.JsonElement mlx = edge.GetProperty("mlx");
        byte[] weight = EngramFixtures.Bytes(mlx.GetProperty("weight"));
        byte[] scales = EngramFixtures.Bytes(mlx.GetProperty("scales"));
        byte[] biases = EngramFixtures.Bytes(mlx.GetProperty("biases"));
        ushort[] expected = EngramFixtures.Bf16(mlx.GetProperty("bf16"));
        MlxAffineRowLayout layout = new(1000, 0, 0, 0);

        for (int r = 0; r < edge.GetProperty("rows").GetInt32(); r++)
        {
            byte[] packed = [.. weight.AsSpan(r * 128, 128), .. scales.AsSpan(r * 16, 16), .. biases.AsSpan(r * 16, 16)];
            ushort[] actual = new ushort[256];
            layout.DecodeRow(packed, actual);
            EngramFixtures.AssertBf16Equal(expected.AsSpan(r * 256, 256), actual, $"mlx edge row {r}");
        }
    }

    [Fact]
    public void Synthetic_AllLayoutsDecodeEveryRowLikePython()
    {
        System.Text.Json.JsonElement synthetic = EngramFixtures.RowDecode.GetProperty("synthetic");
        int rows = synthetic.GetProperty("rows").GetInt32();
        SyntheticTables tables = SyntheticTables.Load(synthetic);
        for (int r = 0; r < rows; r++)
        {
            ushort[] actual = new ushort[256];
            new OfficialFp8E8M0RowLayout(rows, 0, 0).DecodeRow(tables.OfficialPacked(r), actual);
            EngramFixtures.AssertBf16Equal(tables.Expected(SyntheticTables.Kind.Official, r), actual, $"official row {r}");
            new GgufRow264Layout(rows, 0).DecodeRow(tables.GgufPacked(r), actual);
            EngramFixtures.AssertBf16Equal(tables.Expected(SyntheticTables.Kind.Gguf, r), actual, $"gguf row {r}");
            new MlxAffineRowLayout(rows, 0, 0, 0).DecodeRow(tables.MlxPacked(r), actual);
            EngramFixtures.AssertBf16Equal(tables.Expected(SyntheticTables.Kind.Mlx, r), actual, $"mlx row {r}");
        }
    }

    [Fact]
    public void OfficialAndGguf_AgreeOnTheSameCodesAndScales()
    {
        System.Text.Json.JsonElement synthetic = EngramFixtures.RowDecode.GetProperty("synthetic");
        SyntheticTables tables = SyntheticTables.Load(synthetic);
        for (int r = 0; r < synthetic.GetProperty("rows").GetInt32(); r++)
            Assert.Equal(tables.OfficialPacked(r), tables.GgufPacked(r));
    }

    [Fact]
    public void RealRows_DecodeLikePythonOnBytesFetchedFromThePinnedRepos()
    {
        System.Text.Json.JsonElement real = EngramFixtures.RowDecode.GetProperty("real");
        int checkedRows = 0;
        foreach (System.Text.Json.JsonElement row in real.GetProperty("official").EnumerateArray())
        {
            byte[] packed = [.. EngramFixtures.Bytes(row.GetProperty("fp8")), .. EngramFixtures.Bytes(row.GetProperty("e8m0"))];
            ushort[] actual = new ushort[256];
            new OfficialFp8E8M0RowLayout(384006168, 0, 0).DecodeRow(packed, actual);
            EngramFixtures.AssertBf16Equal(EngramFixtures.Bf16(row.GetProperty("bf16")), actual,
                $"official layer {row.GetProperty("layer")} row {row.GetProperty("row")}");
            checkedRows++;
        }
        foreach (System.Text.Json.JsonElement row in real.GetProperty("gguf").EnumerateArray())
        {
            ushort[] actual = new ushort[256];
            new GgufRow264Layout(384006168, 0).DecodeRow(EngramFixtures.Bytes(row.GetProperty("row264")), actual);
            EngramFixtures.AssertBf16Equal(EngramFixtures.Bf16(row.GetProperty("bf16")), actual,
                $"gguf layer {row.GetProperty("layer")} row {row.GetProperty("row")}");
            checkedRows++;
        }
        foreach (System.Text.Json.JsonElement row in real.GetProperty("mlx").EnumerateArray())
        {
            byte[] packed = [.. EngramFixtures.Bytes(row.GetProperty("weight")), .. EngramFixtures.Bytes(row.GetProperty("scales")),
                .. EngramFixtures.Bytes(row.GetProperty("biases"))];
            ushort[] actual = new ushort[256];
            new MlxAffineRowLayout(384006168, 0, 0, 0).DecodeRow(packed, actual);
            EngramFixtures.AssertBf16Equal(EngramFixtures.Bf16(row.GetProperty("bf16")), actual,
                $"mlx layer {row.GetProperty("layer")} row {row.GetProperty("row")}");
            checkedRows++;
        }
        Assert.Equal(12 + 9 + 9, checkedRows);
    }

    [Fact]
    public void RealRows_OfficialAndGgufAreTheSameFp8Bytes()
    {
        // The GGUF conversion copies the official codes and scales verbatim, so layer 1 rows fetched from both must match.
        System.Text.Json.JsonElement real = EngramFixtures.RowDecode.GetProperty("real");
        Dictionary<int, byte[]> official = new();
        foreach (System.Text.Json.JsonElement row in real.GetProperty("official").EnumerateArray())
        {
            if (row.GetProperty("layer").GetInt32() == 1)
                official[row.GetProperty("row").GetInt32()] = [.. EngramFixtures.Bytes(row.GetProperty("fp8")), .. EngramFixtures.Bytes(row.GetProperty("e8m0"))];
        }
        int matched = 0;
        foreach (System.Text.Json.JsonElement row in real.GetProperty("gguf").EnumerateArray())
        {
            if (row.GetProperty("layer").GetInt32() == 1 && official.TryGetValue(row.GetProperty("row").GetInt32(), out byte[]? bytes))
            {
                Assert.Equal(bytes, EngramFixtures.Bytes(row.GetProperty("row264")));
                matched++;
            }
        }
        Assert.True(matched >= 5, $"only {matched} layer-1 rows are present in both the official and GGUF fixtures.");
    }

    [Fact]
    public void Layouts_DescribeTheirPlanes()
    {
        OfficialFp8E8M0RowLayout official = new(1000, 664, 98305579672);
        Assert.Equal(264, official.PackedRowBytes);
        Assert.Equal(new EngramRowSlice(664, 256), official.GetSlice(0));
        Assert.Equal(new EngramRowSlice(98305579672, 8), official.GetSlice(1));

        GgufRow264Layout gguf = new(1000, 4096);
        Assert.Equal(264, gguf.PackedRowBytes);
        Assert.Equal(new EngramRowSlice(4096, 264), gguf.GetSlice(0));

        MlxAffineRowLayout mlx = new(1000, 10, 20, 30);
        Assert.Equal(160, mlx.PackedRowBytes);
        Assert.Equal(new EngramRowSlice(10, 128, 0), mlx.GetSlice(0));
        Assert.Equal(new EngramRowSlice(20, 16, 1), mlx.GetSlice(1));
        Assert.Equal(new EngramRowSlice(30, 16, 2), mlx.GetSlice(2));
    }

    [Fact]
    public void Bf16Rounding_RoundsToNearestEvenAndKeepsSpecials()
    {
        Assert.Equal((ushort)0x3F80, Bf16Rounding.FromSingle(1.0f));
        Assert.Equal((ushort)0x3F80, Bf16Rounding.FromSingle(BitConverter.UInt32BitsToSingle(0x3F808000))); // tie, even stays
        Assert.Equal((ushort)0x3F82, Bf16Rounding.FromSingle(BitConverter.UInt32BitsToSingle(0x3F818000))); // tie, odd rounds up
        Assert.Equal((ushort)0x3F81, Bf16Rounding.FromSingle(BitConverter.UInt32BitsToSingle(0x3F808001)));
        Assert.Equal((ushort)0x7F80, Bf16Rounding.FromSingle(float.PositiveInfinity));
        Assert.Equal((ushort)0x7F80, Bf16Rounding.FromSingle(BitConverter.UInt32BitsToSingle(0x7F7FFFFF))); // max float rounds to +inf
        Assert.True((Bf16Rounding.FromSingle(float.NaN) & 0x7FFF) > 0x7F80);
        Assert.Equal((ushort)0x8000, Bf16Rounding.FromSingle(-0f));
    }
}
