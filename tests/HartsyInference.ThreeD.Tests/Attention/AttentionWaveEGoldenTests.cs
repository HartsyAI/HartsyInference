using HartsyInference.Tests.Common.Attention;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Attention;

/// <summary>Bit-exact regression for the Wave E 3D attention sites: each case in
/// <see cref="AttentionWaveEGoldenCases"/> must reproduce the digests captured from the pre-migration code on the CPU backend
/// (<c>Fixtures/Attention/wave_e_threed.golden.txt</c>).</summary>
public sealed class AttentionWaveEGoldenTests
{
    private static readonly Lazy<Dictionary<string, string>> Golden = new(() => AttentionGolden.LoadFixture("wave_e_threed"));

    public static TheoryData<string> Cases => new(AttentionWaveEGoldenCases.Names);

    [Theory]
    [MemberData(nameof(Cases))]
    public void Site_ReproducesThePreMigrationOutputBitForBit(string name)
    {
        Assert.True(Golden.Value.TryGetValue(name, out string? expected), $"Case '{name}' is missing from the fixture.");
        Assert.Equal(expected, AttentionWaveEGoldenCases.Run(name));
    }

    [Fact]
    public void Fixture_HoldsExactlyTheCases()
    {
        Assert.Equal(AttentionWaveEGoldenCases.Names.Order(StringComparer.Ordinal), Golden.Value.Keys.Order(StringComparer.Ordinal));
    }
}
