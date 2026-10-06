using System.Globalization;
using System.Text;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.Audio.Phonemizer.MeCab;

/// <summary>A pure-C# MeCab (0.996) morphological analyzer over a compiled dictionary directory (<c>sys.dic</c>,
/// <c>unk.dic</c>, <c>matrix.bin</c>, <c>char.bin</c>, optional <c>dicrc</c>). It reproduces MeCab's one-best
/// analysis: dictionary common-prefix lookup, unknown-word candidates per <c>char.def</c> category
/// (invoke/group/length), and the Viterbi search over word plus connection costs, keeping MeCab's node order so
/// ties resolve identically. The dictionary files are memory-mapped, not parsed. Thread-safe.</summary>
internal sealed class MeCabTokenizer : IDisposable
{
    private const int DefaultMaxGroupingSize = 24;
    private const int MaxPrefixResults = 512;
    private const int MaxLookupBytes = 65535;

    private readonly MeCabDictionary _system;
    private readonly MeCabDictionary _unknown;
    private readonly MeCabConnector _matrix;
    private readonly MeCabCharProperty _chars;
    private readonly (int First, int Count)[] _unknownEntries;
    private readonly MeCabCharInfo _space;
    private readonly int _maxGroupingSize;

    private MeCabTokenizer(MeCabDictionary system, MeCabDictionary unknown, MeCabConnector matrix,
        MeCabCharProperty chars, int maxGroupingSize)
    {
        _system = system;
        _unknown = unknown;
        _matrix = matrix;
        _chars = chars;
        _maxGroupingSize = maxGroupingSize;
        _space = chars.Get(0x20);
        _unknownEntries = new (int, int)[chars.Names.Count];
        for (int i = 0; i < _unknownEntries.Length; i++)
        {
            int value = unknown.ExactMatchSearch(Encoding.UTF8.GetBytes(chars.Names[i]));
            if (value < 0)
                throw new HartsyInferenceException($"MeCab unk.dic has no entries for category '{chars.Names[i]}'.");
            _unknownEntries[i] = (value >> 8, value & 0xFF);
        }
    }

    /// <summary>Opens the dictionary in <paramref name="directory"/>.</summary>
    /// <exception cref="FileNotFoundException">A required dictionary file is missing.</exception>
    /// <exception cref="HartsyInferenceException">The files are malformed or do not fit together.</exception>
    public static MeCabTokenizer Open(string directory)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        MeCabCharProperty chars = MeCabCharProperty.Load(Path.Combine(directory, "char.bin"));
        int maxGrouping = ReadMaxGroupingSize(Path.Combine(directory, "dicrc"));
        MeCabDictionary? system = null, unknown = null;
        MeCabConnector? matrix = null;
        try
        {
            system = new MeCabDictionary(Path.Combine(directory, "sys.dic"));
            unknown = new MeCabDictionary(Path.Combine(directory, "unk.dic"));
            matrix = new MeCabConnector(Path.Combine(directory, "matrix.bin"));
            if (system.Type != 0)
                throw new HartsyInferenceException($"'{directory}/sys.dic' is not a system dictionary.");
            if (system.LeftSize != matrix.LeftSize || system.RightSize != matrix.RightSize)
                throw new HartsyInferenceException($"'{directory}': matrix.bin and sys.dic context sizes differ.");
            return new MeCabTokenizer(system, unknown, matrix, chars, maxGrouping);
        }
        catch
        {
            system?.Dispose();
            unknown?.Dispose();
            matrix?.Dispose();
            throw;
        }
    }

    /// <summary>The best-path morphemes of <paramref name="text"/>, as MeCab's <c>parseToNode</c> walks them
    /// (BOS/EOS excluded).</summary>
    public List<MeCabToken> Tokenize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        byte[] sentence = Encoding.UTF8.GetBytes(text);
        int len = sentence.Length;
        Lattice lattice = new(len);
        lattice.Add(new Node { Prev = -1, BNext = -1, ENext = -1 });
        lattice.EndHead[0] = 0;
        Span<(int Value, int Length)> matches = new (int, int)[MaxPrefixResults];
        for (int pos = 0; pos < len; pos++)
        {
            if (lattice.EndHead[pos] < 0) continue;
            int head = Lookup(sentence, pos, lattice, matches);
            Connect(lattice, pos, head);
        }
        int eos = lattice.Add(new Node { Begin = len, Prev = -1, BNext = -1, ENext = -1 });
        for (int pos = len; pos >= 0; pos--)
        {
            if (lattice.EndHead[pos] < 0) continue;
            Connect(lattice, pos, eos);
            break;
        }
        List<MeCabToken> tokens = [];
        for (int n = lattice.Nodes[eos].Prev; n > 0; n = lattice.Nodes[n].Prev)
        {
            ref Node node = ref lattice.Nodes[n];
            MeCabDictionary dic = node.Unknown ? _unknown : _system;
            tokens.Add(new MeCabToken(Encoding.UTF8.GetString(sentence, node.Begin, node.Length),
                dic.Feature(node.Feature), node.CharType, node.Unknown));
        }
        tokens.Reverse();
        return tokens;
    }

    /// <summary>Unmaps the dictionary files.</summary>
    public void Dispose()
    {
        _system.Dispose();
        _unknown.Dispose();
        _matrix.Dispose();
    }

    // MeCab Tokenizer::lookup: the candidate nodes starting at pos, as a bnext-linked list (newest first).
    private int Lookup(byte[] sentence, int pos, Lattice lattice, Span<(int Value, int Length)> matches)
    {
        int end = Math.Min(sentence.Length, pos + MaxLookupBytes);
        int begin2 = pos;
        MeCabCharInfo previous = _space;
        MeCabCharInfo cinfo = default;
        int mblen = 0;
        while (begin2 != end)
        {
            cinfo = _chars.Get(sentence, begin2, end, out mblen);
            if (!previous.IsKindOf(cinfo)) break;
            begin2 += mblen;
            previous = cinfo;
        }
        int head = -1;
        int found = Math.Min(_system.CommonPrefixSearch(sentence.AsSpan(begin2, end - begin2), matches), matches.Length);
        for (int i = 0; i < found; i++)
        {
            (int value, int length) = matches[i];
            int first = value >> 8, count = value & 0xFF;
            for (int j = 0; j < count; j++)
            {
                MeCabEntry entry = _system.Entry(first + j);
                head = lattice.Add(new Node
                {
                    Begin = begin2, Length = length, RLength = begin2 - pos + length, LeftId = entry.LeftId,
                    RightId = entry.RightId, WordCost = entry.Cost, Feature = entry.FeatureOffset,
                    CharType = cinfo.DefaultType, BNext = head, Prev = -1, ENext = -1,
                });
            }
        }
        if (head >= 0 && !cinfo.Invoke) return head;

        int begin3 = begin2 + mblen;
        int groupBegin3 = -1;
        if (begin3 > end)
        {
            head = AddUnknown(lattice, head, cinfo, pos, begin2, begin3);
            if (head >= 0) return head;
        }
        if (cinfo.Group && begin3 <= end)
        {
            int p = begin3;
            int clen = 0;
            MeCabCharInfo c = cinfo;
            while (p < end)
            {
                MeCabCharInfo next = _chars.Get(sentence, p, end, out int step);
                if (!c.IsKindOf(next)) break;
                p += step;
                clen++;
                c = next;
            }
            if (clen <= _maxGroupingSize) head = AddUnknown(lattice, head, cinfo, pos, begin2, p);
            groupBegin3 = p;
        }
        for (int i = 1; i <= cinfo.Length; i++)
        {
            if (begin3 > end) break;
            if (begin3 == groupBegin3) continue;
            head = AddUnknown(lattice, head, cinfo, pos, begin2, begin3);
            if (!cinfo.IsKindOf(_chars.Get(sentence, begin3, end, out mblen))) break;
            begin3 += mblen;
        }
        if (head < 0) head = AddUnknown(lattice, head, cinfo, pos, begin2, begin3);
        return head;
    }

    private int AddUnknown(Lattice lattice, int head, MeCabCharInfo cinfo, int pos, int begin2, int begin3)
    {
        (int first, int count) = _unknownEntries[cinfo.DefaultType];
        for (int k = 0; k < count; k++)
        {
            MeCabEntry entry = _unknown.Entry(first + k);
            head = lattice.Add(new Node
            {
                Begin = begin2, Length = begin3 - begin2, RLength = begin3 - pos, LeftId = entry.LeftId,
                RightId = entry.RightId, WordCost = entry.Cost, Feature = entry.FeatureOffset,
                CharType = cinfo.DefaultType, Unknown = true, BNext = head, Prev = -1, ENext = -1,
            });
        }
        return head;
    }

    // MeCab connect(): each right node takes the cheapest left node ending at pos (first wins ties), then joins the
    // end list at its own end position.
    private void Connect(Lattice lattice, int pos, int head)
    {
        Node[] nodes = lattice.Nodes;
        for (int r = head; r >= 0; r = nodes[r].BNext)
        {
            long bestCost = int.MaxValue;
            int best = -1;
            int leftId = nodes[r].LeftId;
            int wordCost = nodes[r].WordCost;
            for (int l = lattice.EndHead[pos]; l >= 0; l = nodes[l].ENext)
            {
                long cost = nodes[l].Total + _matrix.Cost(nodes[l].RightId, leftId) + wordCost;
                if (cost < bestCost)
                {
                    best = l;
                    bestCost = cost;
                }
            }
            if (best < 0) throw new HartsyInferenceException("MeCab lattice has no path (cost overflow).");
            nodes[r].Prev = best;
            nodes[r].Total = bestCost;
            int x = pos + nodes[r].RLength;
            nodes[r].ENext = lattice.EndHead[x];
            lattice.EndHead[x] = r;
        }
    }

    private static int ReadMaxGroupingSize(string dicrc)
    {
        if (!File.Exists(dicrc)) return DefaultMaxGroupingSize;
        foreach (string raw in File.ReadLines(dicrc))
        {
            int eq = raw.IndexOf('=');
            if (eq < 0 || raw.TrimStart().StartsWith(';')) continue;
            if (raw[..eq].Trim() != "max-grouping-size") continue;
            return int.TryParse(raw[(eq + 1)..].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v)
                && v > 0 ? v : DefaultMaxGroupingSize;
        }
        return DefaultMaxGroupingSize;
    }

    private struct Node
    {
        public int Begin;
        public int Length;
        public int RLength;
        public ushort LeftId;
        public ushort RightId;
        public short WordCost;
        public uint Feature;
        public int CharType;
        public bool Unknown;
        public long Total;
        public int Prev;
        public int BNext;
        public int ENext;
    }

    private sealed class Lattice(int length)
    {
        // Unknown-word candidates may end a few bytes past the sentence (MeCab reads its NUL terminator).
        public int[] EndHead { get; } = NewEndHeads(length + 8);

        public Node[] Nodes { get; private set; } = new Node[Math.Max(16, length * 4)];

        public int Count { get; private set; }

        public int Add(Node node)
        {
            if (Count == Nodes.Length)
            {
                Node[] grown = new Node[Nodes.Length * 2];
                Array.Copy(Nodes, grown, Count);
                Nodes = grown;
            }
            Nodes[Count] = node;
            return Count++;
        }

        private static int[] NewEndHeads(int size)
        {
            int[] heads = new int[size];
            Array.Fill(heads, -1);
            return heads;
        }
    }
}
