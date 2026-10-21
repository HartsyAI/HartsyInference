using Xunit;

namespace HartsyInference.Diffusion.Tests.Attention;

/// <summary>Bit-exact regression for the Wave B attention blocks: each case in <see cref="AttentionWaveBGoldenCases"/> must
/// reproduce the digests captured from the pre-migration attention code on the CPU backend
/// (<c>Fixtures/Attention/wave_b.golden.txt</c>).</summary>
public sealed class AttentionWaveBGoldenTests
{
    private static readonly Lazy<Dictionary<string, string>> Golden = new(Load);

    public static TheoryData<string> Cases => new(AttentionWaveBGoldenCases.Names);

    [Theory]
    [MemberData(nameof(Cases))]
    public void Block_ReproducesThePreMigrationOutputBitForBit(string name)
    {
        Assert.True(Golden.Value.TryGetValue(name, out string? expected), $"Case '{name}' is missing from the fixture.");
        Assert.Equal(expected, AttentionWaveBGoldenCases.Run(name));
    }

    [Fact]
    public void Fixture_HoldsExactlyTheCases()
    {
        Assert.Equal(AttentionWaveBGoldenCases.Names.Order(StringComparer.Ordinal), Golden.Value.Keys.Order(StringComparer.Ordinal));
    }

    private static Dictionary<string, string> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Attention", "wave_b.golden.txt");
        Dictionary<string, string> cases = new(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(path))
        {
            if (line.Length == 0 || line[0] == '#')
                continue;
            int split = line.IndexOf(' ');
            cases.Add(line[..split], line[(split + 1)..]);
        }
        return cases;
    }
}
