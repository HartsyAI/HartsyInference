using System.Text.Json;

namespace HartsyInference.Core.Tests.Engram;

/// <summary>The seeded synthetic tables of the row-decode fixture, in the byte layout of each real format.</summary>
internal sealed class SyntheticTables
{
    public enum Kind { Official, Gguf, Mlx }

    private SyntheticTables(int rows, byte[] fp8, byte[] e8m0, byte[] row264, byte[] weight, byte[] scales, byte[] biases,
        ushort[] official, ushort[] gguf, ushort[] mlx)
    {
        Rows = rows;
        Fp8 = fp8;
        E8m0 = e8m0;
        Row264 = row264;
        MlxWeight = weight;
        MlxScales = scales;
        MlxBiases = biases;
        _official = official;
        _gguf = gguf;
        _mlx = mlx;
    }

    private readonly ushort[] _official;
    private readonly ushort[] _gguf;
    private readonly ushort[] _mlx;

    public int Rows { get; }
    public byte[] Fp8 { get; }
    public byte[] E8m0 { get; }
    public byte[] Row264 { get; }
    public byte[] MlxWeight { get; }
    public byte[] MlxScales { get; }
    public byte[] MlxBiases { get; }

    public static SyntheticTables Load(JsonElement synthetic)
    {
        JsonElement official = synthetic.GetProperty("official");
        JsonElement gguf = synthetic.GetProperty("gguf");
        JsonElement mlx = synthetic.GetProperty("mlx");
        return new SyntheticTables(synthetic.GetProperty("rows").GetInt32(),
            EngramFixtures.Bytes(official.GetProperty("fp8")), EngramFixtures.Bytes(official.GetProperty("e8m0")),
            EngramFixtures.Bytes(gguf.GetProperty("row264")),
            EngramFixtures.Bytes(mlx.GetProperty("weight")), EngramFixtures.Bytes(mlx.GetProperty("scales")), EngramFixtures.Bytes(mlx.GetProperty("biases")),
            EngramFixtures.Bf16(official.GetProperty("bf16")), EngramFixtures.Bf16(gguf.GetProperty("bf16")), EngramFixtures.Bf16(mlx.GetProperty("bf16")));
    }

    public ReadOnlySpan<ushort> Expected(Kind kind, int row) =>
        (kind switch { Kind.Official => _official, Kind.Gguf => _gguf, _ => _mlx }).AsSpan(row * 256, 256);

    public byte[] OfficialPacked(int row) => [.. Fp8.AsSpan(row * 256, 256), .. E8m0.AsSpan(row * 8, 8)];

    public byte[] GgufPacked(int row) => Row264.AsSpan(row * 264, 264).ToArray();

    public byte[] MlxPacked(int row) =>
        [.. MlxWeight.AsSpan(row * 128, 128), .. MlxScales.AsSpan(row * 16, 16), .. MlxBiases.AsSpan(row * 16, 16)];
}
