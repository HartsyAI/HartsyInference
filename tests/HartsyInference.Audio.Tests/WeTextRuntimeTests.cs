using System.Text;
using HartsyInference.Audio.Frontends.WeText;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The OpenFst reader, the kaldifst one-best rewrite and the token reorder, on hand-built inputs. The
/// equal-weight tie cases were checked against kaldifst 1.8.1 <c>TextNormalizer</c> on the same FSTs: OpenFst breaks
/// ties by queue order, which depends on whether the rule is acyclic and whether the composition is weighted.</summary>
public sealed class WeTextRuntimeTests
{
    private const float Inf = float.PositiveInfinity;

    [Fact]
    public void Read_RoundTripsVectorFstBinary()
    {
        StdArc[][] arcs =
        [
            [new StdArc(97, 120, 0.5f, 1), new StdArc(0, 121, 1.25f, 2)],
            [new StdArc(98, 98, 0f, 2)],
            [],
        ];
        StdVectorFst fst = StdVectorFst.Read(new MemoryStream(Serialize(0, [Inf, Inf, 0.75f], arcs)));

        Assert.Equal(0, fst.Start);
        Assert.Equal(3, fst.StateCount);
        Assert.Equal(3, fst.ArcCount);
        Assert.Equal(arcs[0], fst.Arcs(0).ToArray());
        Assert.Equal(arcs[1], fst.Arcs(1).ToArray());
        Assert.Equal(0, fst.Arcs(2).Length);
        Assert.Equal(0.75f, fst.Final(2));
        Assert.Equal(Inf, fst.Final(0));
        Assert.False(fst.IsILabelSorted);
        Assert.True(fst.IsAcyclic);
    }

    [Fact]
    public void Read_ComputesSortednessAndCycles()
    {
        StdVectorFst fst = StdVectorFst.Read(new MemoryStream(Serialize(0, [Inf, 0f],
            [[new StdArc(0, 1, 0f, 1), new StdArc(5, 5, 0f, 1)], [new StdArc(3, 3, 0f, 0)]])));
        Assert.True(fst.IsILabelSorted);
        Assert.False(fst.IsAcyclic);
    }

    [Fact]
    public void Read_RejectsUnsupportedFiles()
    {
        byte[] good = Serialize(0, [0f], [[]]);
        byte[] badMagic = (byte[])good.Clone();
        badMagic[0] ^= 0xFF;
        Assert.Throws<InvalidDataException>(() => StdVectorFst.Read(new MemoryStream(badMagic)));
        Assert.Throws<InvalidDataException>(() => StdVectorFst.Read(new MemoryStream(good[..^4])));
        Assert.Throws<InvalidDataException>(() => StdVectorFst.Read(new MemoryStream(Serialize(0, [0f], [[]], "const"))));
        Assert.Throws<InvalidDataException>(() => StdVectorFst.Read(new MemoryStream(
            Serialize(0, [0f], [[new StdArc(1, 1, 0f, 7)]]))));
    }

    // Rule text in AT&T order "src dst ilabel olabel weight"; final lines "state weight". Input is always "a".
    [Theory]
    [InlineData("0 1 97 120 1|0 1 97 121 1|1 0", "x")]
    [InlineData("0 1 97 120 1|0 2 97 121 1|1 0|2 0", "y")]
    [InlineData("0 1 97 120 0.5|0 2 97 121 0.25|1 0.5|2 0.75", "y")]
    [InlineData("0 1 97 120 0|0 2 97 121 0|1 2 0 112 0|1 3 0 113 0|2 3 0 114 0|3 0", "xq")]
    [InlineData("0 1 97 120 0|0 2 97 121 0|1 2 0 112 0|1 3 0 113 0|2 3 0 114 0|4 4 122 122 0|3 0", "yr")]
    [InlineData("0 1 97 120 0|0 2 97 121 0|1 2 0 112 0|1 3 0 113 0|2 3 0 114 0.5|4 4 122 122 0|3 0", "xq")]
    [InlineData("0 1 0 120 0|0 2 97 121 0|1 2 97 122 0|3 3 122 122 0|2 0", "y")]
    [InlineData("0 1 98 120 0|1 0", "")]
    public void Rewrite_BreaksTiesLikeKaldifst(string rule, string expected)
    {
        Assert.Equal(expected, new FstTextRewriter(Compile(rule)).Rewrite("a"));
    }

    [Fact]
    public void Rewrite_TransducesUtf8BytesAndDropsEpsilonOutput()
    {
        byte[] input = Encoding.UTF8.GetBytes("é");
        byte[] output = Encoding.UTF8.GetBytes("ü!");
        StdVectorFst fst = new(0, [Inf, Inf, Inf, 0f],
        [
            [new StdArc(input[0], output[0], 0f, 1)],
            [new StdArc(input[1], 0, 0f, 2)],
            [new StdArc(0, output[1], 0f, 3)],
            [new StdArc(0, output[2], 0f, 3)],
        ]);
        Assert.Equal("ü", new FstTextRewriter(fst).Rewrite("é"));
        Assert.Throws<ArgumentException>(() => new FstTextRewriter(fst).Rewrite("\0"));
    }

    [Theory]
    [InlineData("en", "money { currency_maj: \"$\" integer_part: \"99\"  fractional_part: \"95\" } w { v: \"each\" }",
        "money { integer_part: \"99\" fractional_part: \"95\" currency_maj: \"$\" } w { v: \"each\" }")]
    [InlineData("en", "date { month: \"March\" day: \"3rd\" year: \", 2026\" }",
        "date { day: \"3rd\" month: \"March\" year: \", 2026\" }")]
    [InlineData("en", "date { preserve_order: \"true\" year: \"2026\" month: \"March\" }",
        "date { preserve_order: \"true\" year: \"2026\" month: \"March\" }")]
    [InlineData("en", "w { v: \"say \\\"hi\\\"\" } p { v: \".\" }", "w { v: \"say \\\"hi\\\"\" } p { v: \".\" }")]
    [InlineData("zh", "date { day: \"6\" month: \"5\" year: \"2024\" } char { name: \"，\" }",
        "date { year: \"2024\" month: \"5\" day: \"6\" } char { name: \"，\" }")]
    [InlineData("zh", "money { currency: \"元\" value: \"199.5\" extra: \"x\" }",
        "money { value: \"199.5\" currency: \"元\" }")]
    [InlineData("en", "  cardinal { integer: \"5\" }   ", "cardinal { integer: \"5\" }")]
    public void Reorder_MatchesWetextTokenParser(string language, string tagged, string expected)
    {
        WeTextLanguage lang = language == "zh" ? WeTextLanguage.Chinese : WeTextLanguage.English;
        Assert.Equal(expected, WeTextTokenParser.Reorder(tagged, lang));
    }

    [Theory]
    [InlineData("")]
    [InlineData("w { v: \"unterminated }")]
    [InlineData("w { v: \"ends with escape\\")]
    [InlineData("w {\tv: \"x\" }")]
    public void Reorder_ThrowsWherePythonAssertsOrLoops(string tagged)
    {
        Assert.Throws<FormatException>(() => WeTextTokenParser.Reorder(tagged, WeTextLanguage.English));
    }

    private static StdVectorFst Compile(string rule)
    {
        List<(int Src, StdArc Arc)> arcs = [];
        Dictionary<int, float> finals = [];
        int states = 0;
        foreach (string line in rule.Split('|'))
        {
            string[] f = line.Split(' ');
            int src = int.Parse(f[0]);
            states = Math.Max(states, src + 1);
            if (f.Length == 2)
            {
                finals[src] = float.Parse(f[1], System.Globalization.CultureInfo.InvariantCulture);
                continue;
            }
            StdArc arc = new(int.Parse(f[2]), int.Parse(f[3]),
                float.Parse(f[4], System.Globalization.CultureInfo.InvariantCulture), int.Parse(f[1]));
            states = Math.Max(states, arc.NextState + 1);
            arcs.Add((src, arc));
        }
        float[] final = new float[states];
        List<StdArc>[] perState = new List<StdArc>[states];
        for (int s = 0; s < states; s++)
        {
            final[s] = finals.TryGetValue(s, out float w) ? w : Inf;
            perState[s] = [];
        }
        foreach ((int src, StdArc arc) in arcs) perState[src].Add(arc);
        return new StdVectorFst(0, final, perState);
    }

    // OpenFst binary layout: header (magic, type, arc type, version, flags, properties, start, states, arcs), then per
    // state the final weight, the arc count and each arc as ilabel, olabel, weight, nextstate.
    private static byte[] Serialize(long start, float[] finals, StdArc[][] arcs, string fstType = "vector")
    {
        MemoryStream stream = new();
        using (BinaryWriter writer = new(stream, Encoding.ASCII, leaveOpen: true))
        {
            writer.Write(2125659606);
            writer.Write(fstType.Length);
            writer.Write(Encoding.ASCII.GetBytes(fstType));
            writer.Write("standard".Length);
            writer.Write(Encoding.ASCII.GetBytes("standard"));
            writer.Write(2);
            writer.Write(0);
            writer.Write(3UL);
            writer.Write(start);
            writer.Write((long)finals.Length);
            writer.Write(0L);
            for (int s = 0; s < finals.Length; s++)
            {
                writer.Write(finals[s]);
                writer.Write((long)arcs[s].Length);
                foreach (StdArc arc in arcs[s])
                {
                    writer.Write(arc.ILabel);
                    writer.Write(arc.OLabel);
                    writer.Write(arc.Weight);
                    writer.Write(arc.NextState);
                }
            }
        }
        return stream.ToArray();
    }
}
