using System.Globalization;
using System.Text;
using HartsyInference.Core.Backends;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Attention;

/// <summary>The INT8 attention v2 accuracy table: each variant against the contract oracle on the Phase-0 shapes (sampled heads
/// and rows) and on the adversarial cases. Prints markdown; asserts the base variant meets CK's bar on CK's own uniform method
/// and that no variant but F16 V overflows.</summary>
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
        List<(string Case, Int8V2Spec[] Specs)> realistic = [], adversarial = [];
        foreach ((string Id, int Heads, int Sq, int Skv, int HeadDim, bool Causal) shape in Int8V2Shapes.Phase0)
            realistic.Add(($"{shape.Id} N(0,1)", Int8V2Shapes.HeadsOf(shape, Int8V2Fill.Gaussian).ToArray()));
        realistic.Add(("ck-parity-4k U[-1,1)", Int8V2Shapes.HeadsOf(Int8V2Shapes.Phase0.Single(s => s.Id == "ck-parity-4k"), Int8V2Fill.Uniform)
            .ToArray()));
        foreach (Int8V2Spec spec in Int8V2Adversarial.All()) adversarial.Add((spec.Id, [spec]));
        Dictionary<(string Case, int Variant), (double RelL2, double Cos, double MaxAbs)> measured = [];
        output.WriteLine(Table(realistic, measured));
        output.WriteLine(Table(adversarial, measured));

        // The uniform control is CK's own method: the base variant meets its bar there, and only F16 V overflows anywhere.
        (double rel, double cos, _) = measured[("ck-parity-4k U[-1,1)", 0)];
        Assert.True(rel <= Int8V2Measure.BarRelL2 && cos >= Int8V2Measure.BarCos, $"base on the uniform control: {rel:E2} / {cos:F6}");
        int f16 = Array.FindIndex(Variants, v => v.Options.Probabilities == Int8ProbabilityFormat.F16);
        foreach (((string name, int variant), (double r, _, _)) in measured)
            Assert.True(double.IsFinite(r) == !(name == "v-above-f16-max" && variant == f16), $"{name} / {Variants[variant].Name}: relL2 {r}");
    }

    private static string Table(List<(string Case, Int8V2Spec[] Specs)> groups,
        Dictionary<(string Case, int Variant), (double RelL2, double Cos, double MaxAbs)> measured)
    {
        StringBuilder md = new();
        md.Append("| case (relL2 / cos / max abs) |");
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
                (double rel, double cos, double max) m = Int8V2Measure.Of(actual[v].ToArray(), expected.ToArray());
                measured[(name, v)] = m;
                md.Append(CultureInfo.InvariantCulture, $" {m.rel:0.0e0} / {m.cos:F6} / {m.max:0.0e0} |");
            }
            md.AppendLine();
        }
        return md.ToString();
    }
}
