using HartsyInference.Core.Engram;
using Xunit;

namespace HartsyInference.Core.Tests.Engram;

/// <summary>Row decode of the three storage layouts against python dequantization (fixtures dumped by dump_engram_rows_fixtures.py).</summary>
public sealed class EngramRowLayoutTests
{
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
