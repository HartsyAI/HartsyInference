using Xunit;

namespace HartsyInference.Diffusion.Tests.Attention;

/// <summary>Bit-exact regression for the Wave C attention code (Wan, S2V, MG3, Oasis, VAE, UNet, ControlNet-Union, IP-Adapter, SeedVR2, H3, Cosmos AR): each case in <see cref="AttentionWaveCGoldenCases"/> must
/// reproduce the digests captured from the pre-migration attention code on the CPU backend
/// (<c>Fixtures/Attention/wave_c.golden.txt</c>).</summary>
public sealed class AttentionWaveCGoldenTests
{
    private static readonly Lazy<Dictionary<string, string>> Golden = new(Load);

    public static TheoryData<string> Cases => new(AttentionWaveCGoldenCases.Names);

    [Theory]
    [MemberData(nameof(Cases))]
    public void Block_ReproducesThePreMigrationOutputBitForBit(string name)
    {
        Assert.True(Golden.Value.TryGetValue(name, out string? expected), $"Case '{name}' is missing from the fixture.");
        Assert.Equal(expected, AttentionWaveCGoldenCases.Run(name));
    }

    [Fact]
    public void Fixture_HoldsExactlyTheCases()
    {
        Assert.Equal(AttentionWaveCGoldenCases.Names.Order(StringComparer.Ordinal), Golden.Value.Keys.Order(StringComparer.Ordinal));
    }

    private static Dictionary<string, string> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Attention", "wave_c.golden.txt");
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
