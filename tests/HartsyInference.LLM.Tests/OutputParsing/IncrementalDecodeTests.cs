using System.Text;
using HartsyInference.LLM.OutputParsing;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>The windowed fallback for tokenizers without per-token bytes: concatenated deltas equal a one-shot decode for any id sequence, a partial character is held back until it completes, and no call decodes more than a short window (the O(n^2) re-decode is gone).</summary>
public sealed class IncrementalDecodeTests
{
    private const string Sample = "héllo wörld 日本語 😀 done. � kept! ";

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(23)]
    public void RandomByteSplitsConcatenateToTheFullDecodeWithABoundedWindow(int seed)
    {
        ByteWindowTokenizer tok = new();
        string text = string.Concat(Enumerable.Repeat(Sample, 40));
        int[] ids = tok.RandomSplit(text, new Random(seed), maxPiece: 5);
        Assert.Equal(text, Run(tok, ids));
        // Context (the tokens emitted at the last pivot) plus the tokens held for one unfinished character.
        Assert.True(tok.MaxIdsPerDecode <= 12, $"window grew to {tok.MaxIdsPerDecode} ids");
        Assert.True(tok.DecodeCalls >= ids.Length, "every push must decode at most a window, never the whole history");
    }

    [Fact]
    public void PartialCharacterIsHeldBackUntilTheNextTokenCompletesIt()
    {
        ByteWindowTokenizer tok = new();
        int[] ids = tok.SplitBytes("a😀b", 1, 2, 3, 4, 5);
        IncrementalDetokenizer d = new(tok, includeSpecial: false);
        Assert.Equal("a", d.Push(ids[0]));
        Assert.Equal("", d.Push(ids[1]));
        Assert.Equal("", d.Push(ids[2]));
        Assert.Equal("", d.Push(ids[3]));
        Assert.Equal("😀", d.Push(ids[4]));
        Assert.Equal("b", d.Push(ids[5]));
        Assert.Equal("", d.Flush());
    }

    [Fact]
    public void UnfinishedSequenceAtTheEndFlushesLikeAOneShotDecode()
    {
        ByteWindowTokenizer tok = new();
        int[] ids = [tok.Add("x"u8.ToArray()), tok.Add([0xE6, 0x97])];
        IncrementalDetokenizer d = new(tok, includeSpecial: false);
        Assert.Equal("x", d.Push(ids[0]));
        Assert.Equal("", d.Push(ids[1]));
        Assert.Equal("�", d.Flush());
        Assert.Equal("x�", tok.Decode(ids));
    }

    [Fact]
    public void FirstTokenDummyPrefixStrippingIsCancelledByTheContextToken()
    {
        // SentencePiece-style decoders drop the leading space of the FIRST token they decode; a windowed decode
        // that restarted at every pivot would drop one space per pivot without the emitted context token.
        PrefixStripTokenizer tok = new();
        int[] ids = [.. Enumerable.Range(0, 200).Select(i => i % tok.Words.Length)];
        // The one-shot reference comes from a separate instance so it does not count toward the window statistic.
        Assert.Equal(new PrefixStripTokenizer().Decode(ids), Run(tok, ids));
        Assert.True(tok.MaxIdsPerDecode <= 3, $"window grew to {tok.MaxIdsPerDecode} ids");
    }

    [Fact]
    public void DecoderThatNormalizesAcrossTheBoundaryNeverDropsText()
    {
        // NFC-merging decode breaks the prefix invariant ("e" then U+0301 becomes "é"); slicing at the common prefix
        // may repeat the merged character but must never lose one, and must not throw.
        CombiningMergeTokenizer tok = new();
        int[] ids = [0, 1, 2, 0, 1, 2];
        string streamed = Run(tok, ids);
        string oneShot = new CombiningMergeTokenizer().Decode(ids);
        Assert.True(IsSubsequence(oneShot, streamed), $"streamed '{streamed}' lost characters of '{oneShot}'");
        Assert.EndsWith("x", streamed);
    }

    private static bool IsSubsequence(string needle, string hay)
    {
        int i = 0;
        foreach (char c in hay)
            if (i < needle.Length && needle[i] == c) i++;
        return i == needle.Length;
    }

    private static string Run(IncrementalDetokenizer d, int[] ids)
    {
        StringBuilder sb = new();
        foreach (int id in ids) sb.Append(d.Push(id));
        sb.Append(d.Flush());
        return sb.ToString();
    }

    private static string Run(NoBytesTokenizer tok, int[] ids) => Run(new IncrementalDetokenizer(tok, includeSpecial: false), ids);

    /// <summary>Byte-concat decode without <c>TokenBytes</c>; counts the widest window any decode call saw.</summary>
    private sealed class ByteWindowTokenizer : NoBytesTokenizer
    {
        private readonly List<byte[]> _pieces = [];

        public int MaxIdsPerDecode { get; private set; }

        public int DecodeCalls { get; private set; }

        public int Add(byte[] bytes)
        {
            _pieces.Add(bytes);
            return _pieces.Count - 1;
        }

        public int[] SplitBytes(string text, params int[] points)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(text);
            List<int> ids = [];
            int start = 0;
            foreach (int requested in points.Append(bytes.Length))
            {
                int point = Math.Min(requested, bytes.Length);
                if (point > start) ids.Add(Add(bytes[start..point]));
                start = Math.Max(start, point);
            }
            return [.. ids];
        }

        public int[] RandomSplit(string text, Random random, int maxPiece)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(text);
            List<int> ids = [];
            int pos = 0;
            while (pos < bytes.Length)
            {
                int len = Math.Min(bytes.Length - pos, random.Next(1, maxPiece + 1));
                ids.Add(Add(bytes[pos..(pos + len)]));
                pos += len;
            }
            return [.. ids];
        }

        public override string Decode(IReadOnlyList<int> ids)
        {
            DecodeCalls++;
            MaxIdsPerDecode = Math.Max(MaxIdsPerDecode, ids.Count);
            List<byte> all = [];
            foreach (int id in ids) all.AddRange(_pieces[id]);
            return Encoding.UTF8.GetString([.. all]);
        }
    }

    /// <summary>Pieces "e", U+0301 (combining acute) and "x"; the decoded string is NFC-normalized, so a window that starts on the combining mark merges differently from one that starts on the base letter.</summary>
    private sealed class CombiningMergeTokenizer : NoBytesTokenizer
    {
        private static readonly string[] Pieces = ["e", "́", "x"];

        public override string Decode(IReadOnlyList<int> ids)
        {
            StringBuilder sb = new();
            foreach (int id in ids) sb.Append(Pieces[id]);
            return sb.ToString().Normalize(NormalizationForm.FormC);
        }
    }

    /// <summary>Word pieces carrying a leading space marker; the decoded string drops the marker of its first piece.</summary>
    private sealed class PrefixStripTokenizer : NoBytesTokenizer
    {
        public string[] Words { get; } = ["▁the", "▁quick", "▁brown", "▁fox", ","];

        public int MaxIdsPerDecode { get; private set; }

        public override string Decode(IReadOnlyList<int> ids)
        {
            MaxIdsPerDecode = Math.Max(MaxIdsPerDecode, ids.Count);
            StringBuilder sb = new();
            foreach (int id in ids) sb.Append(Words[id]);
            string text = sb.Replace('▁', ' ').ToString();
            return text.Length > 0 && text[0] == ' ' ? text[1..] : text;
        }
    }
}
