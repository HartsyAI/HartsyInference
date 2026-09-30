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

    [Fact]
    public void FixtureHasEveryRequiredEdgeCase()
    {
        HashSet<string> names = HashFixture().GetProperty("cases").EnumerateArray().Select(c => c.GetProperty("name").GetString()!).ToHashSet();
        foreach (string required in new[] { "single_token_prefill", "dead_middle_span", "dead_first_and_last", "all_dead", "prefill_then_decode_1",
            "prefill_then_decode_dead", "decode_from_single_token_prefill", "chunked_prefill_random", "long_random_masked", "vocab_extremes" })
            Assert.Contains(required, names);
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
    public void Hash_RewindingStartPosRecomputesFromTheEarlierHistory()
    {
        JsonElement testCase = HashFixture().GetProperty("cases").EnumerateArray().Single(c => c.GetProperty("name").GetString() == "prefill_plain");
        JsonElement step = testCase.GetProperty("steps")[0];
        int[] ids = step.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        long[] expected = step.GetProperty("hash").EnumerateArray().Select(e => e.GetInt64()).ToArray();
        EngramHasher hasher = new();
        long[] first = new long[expected.Length];
        hasher.Hash(ids, [], 0, first);
        // A second prefill from 0 must not see the first run's history as its own past.
        long[] second = new long[expected.Length];
        hasher.Hash(ids, [], 0, second);
        Assert.Equal(expected, second);
        Assert.Equal(ids.Length, hasher.Length);
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
    public void Constants_ArePinnedToTheCheckpointRevisionAndHashMatchTheManifest()
    {
        EngramConstants constants = EngramConstants.Default;
        Assert.Equal(PinnedRevision, constants.Revision);
        Assert.Equal("deepseek-ai/DeepSeek-V4.1-Flash", constants.Checkpoint);
        Assert.Equal([1, 14], constants.LayerIds);
        Assert.Equal([384006168L, 384016682L], constants.TableRows);
        Assert.Equal(129280, constants.TokenizerVocab);
        Assert.Equal(99092, constants.CompressedVocab);
        Assert.Equal(2, constants.PadCompressedId);
        Assert.Equal(4, constants.MaxNgramSize);
        Assert.Equal(8, constants.HeadCount);
        Assert.Equal(2 * 4, constants.Multipliers.Length);
        Assert.Equal(2 * 24, constants.Offsets.Length);
        Assert.Equal(2 * 3 * 8, constants.Primes.Length);
    }

    [Fact]
    public void Constants_ManifestRecordsEveryExternalCrossCheckAsPassed()
    {
        string manifestPath = Path.Combine(RepoRoot.Path, "src", "HartsyInference.LLM", "DeepSeekV41", "Engram", "Constants", "manifest.json");
        JsonElement crossChecks = JsonDocument.Parse(File.ReadAllText(manifestPath)).RootElement.GetProperty("cross_checks");
        int passed = 0;
        foreach (JsonProperty source in crossChecks.EnumerateObject())
        {
            Assert.False(string.IsNullOrEmpty(source.Value.GetProperty("revision").GetString()), $"{source.Name} has no pinned revision.");
            foreach (JsonProperty check in source.Value.EnumerateObject())
            {
                if (check.Value.ValueKind == JsonValueKind.String)
                    continue;
                Assert.True(check.Value.ValueKind == JsonValueKind.True, $"cross-check {source.Name}.{check.Name} is not recorded as passed.");
                passed++;
            }
        }
        Assert.Equal(4, passed); // GGUF token map + primes + multipliers, MLX token map
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
