using Xunit;

namespace HartsyInference.Diffusion.Tests.Rope;

/// <summary>Bit-exact regression for every Diffusion RoPE family on the shared Core tables: each case in
/// <see cref="RopeFamilyGoldenCases"/> must reproduce the digests captured from the pre-migration implementations
/// (<c>Fixtures/Rope/family_tables.golden.txt</c>).</summary>
public sealed class RopeFamilyGoldenTests
{
    private static readonly Lazy<Dictionary<string, string>> Golden = new(Load);

    public static TheoryData<string> Cases => new(RopeFamilyGoldenCases.Names);

    [Theory]
    [MemberData(nameof(Cases))]
    public void Family_ReproducesThePreMigrationTablesBitForBit(string name)
    {
        Assert.True(Golden.Value.TryGetValue(name, out string? expected), $"Case '{name}' is missing from the fixture.");
        Assert.Equal(expected, RopeFamilyGoldenCases.Run(name));
    }

    [Fact]
    public void Fixture_HoldsExactlyTheCases()
    {
        Assert.Equal(RopeFamilyGoldenCases.Names.Order(StringComparer.Ordinal), Golden.Value.Keys.Order(StringComparer.Ordinal));
    }

    private static Dictionary<string, string> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Rope", "family_tables.golden.txt");
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
