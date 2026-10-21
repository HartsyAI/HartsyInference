using Xunit;

namespace HartsyInference.LLM.Tests.Attention;

/// <summary>Bit-exact regression for the LLM attention code (GenericTransformer eager, batched and graph decode, MLA, tensor parallel, looped nGPT, Qwen3.5, Mllama cross-attention): each case in
/// <see cref="AttentionWaveDGoldenCases"/> must reproduce the digests captured from the pre-migration attention code on the CPU backend (<c>Fixtures/Attention/wave_d.golden.txt</c>).</summary>
public sealed class AttentionWaveDGoldenTests
{
    private static readonly Lazy<Dictionary<string, string>> Golden = new(Load);

    public static TheoryData<string> Cases => new(AttentionWaveDGoldenCases.Names);

    [Theory]
    [MemberData(nameof(Cases))]
    public void Case_ReproducesThePreMigrationOutputBitForBit(string name)
    {
        Assert.True(Golden.Value.TryGetValue(name, out string? expected), $"Case '{name}' is missing from the fixture.");
        Assert.Equal(expected, AttentionWaveDGoldenCases.Run(name));
    }

    public static TheoryData<string> GraphCases => new(AttentionWaveDGoldenCases.Names.Where(name => name.StartsWith("gt_graph_", StringComparison.Ordinal)));

    /// <summary>Guards the shim itself: a captured-step token stream must equal the eager forward's on the same backend, or the graph digests would pin a shim artifact.</summary>
    [Theory]
    [MemberData(nameof(GraphCases))]
    public void GraphShim_DecodesTheSameStreamAsTheEagerForward(string name)
    {
        (int[] graph, int[] eager) = AttentionWaveDGoldenCases.GraphAndEagerTokens(name);
        Assert.Equal(eager, graph);
    }

    [Fact]
    public void Fixture_HoldsExactlyTheCases()
    {
        Assert.Equal(AttentionWaveDGoldenCases.Names.Order(StringComparer.Ordinal), Golden.Value.Keys.Order(StringComparer.Ordinal));
    }

    private static Dictionary<string, string> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Attention", "wave_d.golden.txt");
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
