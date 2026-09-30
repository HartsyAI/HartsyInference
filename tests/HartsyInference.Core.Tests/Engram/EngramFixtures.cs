using System.Text.Json;
using HartsyInference.Tests.Common;

namespace HartsyInference.Core.Tests.Engram;

/// <summary>Loads the committed Engram row-decode fixture dumped by tests/python-reference/deepseek_v41/dump_engram_rows_fixtures.py.</summary>
internal static class EngramFixtures
{
    public const int Dim = 256;

    private static readonly Lazy<JsonElement> Rows = new(() =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures",
            "engram_row_decode.json"))).RootElement);

    public static JsonElement RowDecode => Rows.Value;

    public static byte[] Bytes(JsonElement e) => Convert.FromBase64String(e.GetString()!);

    public static ushort[] Bf16(JsonElement e)
    {
        byte[] raw = Bytes(e);
        ushort[] values = new ushort[raw.Length / 2];
        Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
        return values;
    }

    /// <summary>Bit-exact bf16 equality, except that any NaN equals any NaN (a python cast and a C# cast may pick different NaN payloads).</summary>
    public static void AssertBf16Equal(ReadOnlySpan<ushort> expected, ReadOnlySpan<ushort> actual, string what)
    {
        Xunit.Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            bool bothNaN = IsNaN(expected[i]) && IsNaN(actual[i]);
            if (!bothNaN && expected[i] != actual[i])
                Xunit.Assert.Fail($"{what}: element {i} expected 0x{expected[i]:X4} but was 0x{actual[i]:X4}.");
        }
    }

    private static bool IsNaN(ushort bits) => (bits & 0x7FFF) > 0x7F80;
}
