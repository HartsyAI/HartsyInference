using System.Globalization;
using System.Text;
using HartsyInference.Core.Backends;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Attention;

/// <summary>The INT8 attention v2 accuracy table: each variant against the contract oracle on the Phase-0 shapes (sampled heads
/// and rows) and on the adversarial cases. Prints markdown; asserts only that every variant ran.</summary>
[Trait("Category", "Slow")]
public sealed class Int8AttentionV2AccuracyTableTests(ITestOutputHelper output)
{
    /// <summary>The base variant and one change at a time.</summary>
    public static readonly (string Name, Int8AttentionV2Options Options)[] Variants =
    [
        ("base: class K, BC128, smooth, u8", new Int8AttentionV2Options()),
        ("per-key K", new Int8AttentionV2Options { KeyScales = Int8KeyScaleGrouping.PerKey }),
        ("BC64", new Int8AttentionV2Options { BlockCols = 64 }),
        ("no smoothing", new Int8AttentionV2Options { SmoothKeys = false }),
        ("per-tile V", new Int8AttentionV2Options { ValueScalesPerTile = true }),
        ("u8 stochastic", new Int8AttentionV2Options { Probabilities = Int8ProbabilityFormat.U8Stochastic, RoundingSeed = 7 }),
        ("fp8 P×V", new Int8AttentionV2Options { Probabilities = Int8ProbabilityFormat.Fp8 }),
        ("f16 P×V", new Int8AttentionV2Options { Probabilities = Int8ProbabilityFormat.F16 }),
    ];

    [Fact]
    public void PrintsTheTable()
    {
        List<(string Case, Int8V2Spec[] Specs)> groups = [];
        foreach ((string Id, int Heads, int Sq, int Skv, int HeadDim, bool Causal) shape in Int8V2Shapes.Phase0)
            groups.Add(($"{shape.Id} gauss", Int8V2Shapes.HeadsOf(shape, Int8V2Fill.Gaussian).ToArray()));
        groups.Add(("ck-parity-4k uniform", Int8V2Shapes.HeadsOf(Int8V2Shapes.Phase0[^2], Int8V2Fill.Uniform).ToArray()));
        foreach (Int8V2Spec spec in Int8V2Adversarial.All()) groups.Add((spec.Id, [spec]));

        StringBuilder md = new();
        md.Append("| case |");
        foreach ((string name, _) in Variants) md.Append(CultureInfo.InvariantCulture, $" {name} |");
        md.AppendLine().Append("|---|").Append(string.Concat(Enumerable.Repeat("---|", Variants.Length))).AppendLine();
        foreach ((string name, Int8V2Spec[] specs) in groups)
        {
            List<float>[] actual = Variants.Select(_ => new List<float>()).ToArray();
            List<float> expected = [];
            foreach (Int8V2Spec spec in specs)
            {
                using Int8V2Case c = Int8V2Case.Create(spec);
                expected.AddRange(c.Oracle());
                for (int v = 0; v < Variants.Length; v++) actual[v].AddRange(c.Emulate(Variants[v].Options));
            }
            md.Append(CultureInfo.InvariantCulture, $"| {name} |");
            for (int v = 0; v < Variants.Length; v++)
            {
                (double rel, double cos, double max) = Int8V2Measure.Of(actual[v].ToArray(), expected.ToArray());
                md.Append(CultureInfo.InvariantCulture, $" {rel:E1} / {cos:F6} / {max:E1} |");
            }
            md.AppendLine();
            output.WriteLine(md.ToString().Split('\n')[^2]);
        }
        output.WriteLine("");
        output.WriteLine(md.ToString());
        Assert.True(groups.Count > 0);
    }
}
