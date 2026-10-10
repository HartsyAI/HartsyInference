using System.Text.Json;
using HartsyInference.LLM.DeepSeekV41.Engram;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41.Engram;

/// <summary>The C# Engram hash against ids dumped from the unmodified upstream <c>engram.py</c> (dump_engram_hash_fixtures.py).</summary>
public sealed class EngramHasherTests
{
    private const string PinnedRevision = "dba1be0a40aa45a94ad051997016db3960a90277";
    private static readonly string FixtureDir = Path.Combine(RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures");

    private static JsonElement HashFixture() => JsonDocument.Parse(File.ReadAllText(Path.Combine(FixtureDir, "engram_hash_ids.json"))).RootElement;

    public static IEnumerable<object[]> CaseNames()
    {
        foreach (JsonElement c in HashFixture().GetProperty("cases").EnumerateArray())
            yield return [c.GetProperty("name").GetString()!];
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Hash_EqualsUpstreamPythonExactly(string caseName)
    {
        JsonElement fixture = HashFixture();
        Assert.Equal(PinnedRevision, fixture.GetProperty("checkpointRevision").GetString());
        JsonElement testCase = fixture.GetProperty("cases").EnumerateArray().Single(c => c.GetProperty("name").GetString() == caseName);
        EngramHasher hasher = new();
        int step = 0;
        foreach (JsonElement s in testCase.GetProperty("steps").EnumerateArray())
        {
            int startPos = s.GetProperty("startPos").GetInt32();
            int[] ids = s.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            bool[] mask = s.GetProperty("mask").ValueKind == JsonValueKind.Null
                ? []
                : s.GetProperty("mask").EnumerateArray().Select(e => e.GetInt32() != 0).ToArray();
            long[] expected = s.GetProperty("hash").EnumerateArray().Select(e => e.GetInt64()).ToArray();
            long[] actual = new long[ids.Length * hasher.ValuesPerPosition];
            hasher.Hash(ids, mask, startPos, actual);
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                if (expected[i] != actual[i])
                    Assert.Fail($"{caseName} step {step}: position {i / hasher.ValuesPerPosition} value {i % hasher.ValuesPerPosition} expected {expected[i]} but was {actual[i]}.");
            }
            step++;
        }
    }

    [Fact]
    public void Hash_RejectsGapsBadMasksAndOutOfRangeIds()
    {
        EngramHasher hasher = new();
        long[] dest = new long[hasher.ValuesPerPosition * 4];
        Assert.Throws<ArgumentOutOfRangeException>(() => hasher.Hash([1, 2], [], 1, dest));
        Assert.Throws<ArgumentException>(() => hasher.Hash([1, 2], [true], 0, dest));
        Assert.Throws<ArgumentOutOfRangeException>(() => hasher.Hash([EngramConstants.Default.TokenizerVocab], [], 0, dest));
        Assert.Throws<ArgumentOutOfRangeException>(() => hasher.Hash([-1], [], 0, dest));
        Assert.Throws<ArgumentException>(() => hasher.Hash([1, 2], [], 0, new long[hasher.ValuesPerPosition]));
    }

    [Fact]
    public void Hash_DeadTokenAndSequenceStartMapToThePadRow()
    {
        // A one-token prefill only sees itself: every lookback past the start is the pad id, so order >= 2 rows for
        // a masked token equal those of an all-pad window, independent of the token id.
        EngramHasher a = new();
        EngramHasher b = new();
        long[] x = new long[a.ValuesPerPosition];
        long[] y = new long[a.ValuesPerPosition];
        a.Hash([10000], [false], 0, x);
        b.Hash([90000], [false], 0, y);
        Assert.Equal(x, y);
    }

    [Fact]
    public void Constants_OffsetsAreTheRunningSumOfPrimesAndTablesSumToRows()
    {
        EngramConstants constants = EngramConstants.Default;
        ReadOnlySpan<long> primes = constants.Primes;
        ReadOnlySpan<long> offsets = constants.Offsets;
        for (int layer = 0; layer < 2; layer++)
        {
            long running = 0;
            for (int col = 0; col < 24; col++)
            {
                Assert.Equal(running, offsets[layer * 24 + col]);
                running += primes[layer * 24 + col];
            }
            Assert.Equal(constants.TableRows[layer], running);
        }
    }
}
