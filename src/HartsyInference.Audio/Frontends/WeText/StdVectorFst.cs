using System.Text;

namespace HartsyInference.Audio.Frontends.WeText;

/// <summary>An immutable OpenFst <c>VectorFst&lt;StdArc&gt;</c> read from its binary file: per-state final weights and
/// arcs in file order. Input-label sortedness and acyclicity are computed on load, as OpenFst does when it converts a
/// loaded vector FST to the <c>ConstFst</c> kaldifst composes against; both decide composition and queue order.</summary>
public sealed class StdVectorFst
{
    private const int Magic = 2125659606;
    private const int MinVectorVersion = 2;
    private const int HasInputSymbols = 1;
    private const int HasOutputSymbols = 2;

    private readonly float[] _finals;
    private readonly int[] _arcOffsets;
    private readonly StdArc[] _arcs;

    /// <summary>Builds an FST from per-state finals (<see cref="float.PositiveInfinity"/> for non-final) and arcs.</summary>
    public StdVectorFst(int start, IReadOnlyList<float> finals, IReadOnlyList<IReadOnlyList<StdArc>> arcs)
    {
        ArgumentNullException.ThrowIfNull(finals);
        ArgumentNullException.ThrowIfNull(arcs);
        if (finals.Count != arcs.Count) throw new ArgumentException("finals and arcs must have one entry per state");
        int states = finals.Count;
        if (start < -1 || start >= states) throw new ArgumentOutOfRangeException(nameof(start));
        Start = start;
        _finals = new float[states];
        _arcOffsets = new int[states + 1];
        int total = 0;
        for (int s = 0; s < states; s++) total += arcs[s].Count;
        _arcs = new StdArc[total];
        int next = 0;
        for (int s = 0; s < states; s++)
        {
            _finals[s] = finals[s];
            _arcOffsets[s] = next;
            foreach (StdArc arc in arcs[s])
            {
                if (arc.NextState < 0 || arc.NextState >= states)
                    throw new ArgumentException($"arc from state {s} targets missing state {arc.NextState}");
                _arcs[next++] = arc;
            }
        }
        _arcOffsets[states] = next;
        IsILabelSorted = ComputeILabelSorted();
        IsAcyclic = ComputeAcyclic();
    }

    private StdVectorFst(int start, float[] finals, int[] arcOffsets, StdArc[] arcs)
    {
        Start = start;
        _finals = finals;
        _arcOffsets = arcOffsets;
        _arcs = arcs;
        IsILabelSorted = ComputeILabelSorted();
        IsAcyclic = ComputeAcyclic();
    }

    /// <summary>The start state, or -1 for the empty FST.</summary>
    public int Start { get; }

    /// <summary>Number of states.</summary>
    public int StateCount => _finals.Length;

    /// <summary>Total number of arcs.</summary>
    public int ArcCount => _arcs.Length;

    /// <summary>True when every state's arcs are in non-decreasing input-label order.</summary>
    public bool IsILabelSorted { get; }

    /// <summary>True when no cycle exists anywhere in the FST.</summary>
    public bool IsAcyclic { get; }

    /// <summary>Final weight of <paramref name="state"/>; <see cref="float.PositiveInfinity"/> (tropical zero) if not
    /// final.</summary>
    public float Final(int state) => _finals[state];

    /// <summary>The arcs leaving <paramref name="state"/>, in file order.</summary>
    public ReadOnlySpan<StdArc> Arcs(int state) =>
        _arcs.AsSpan(_arcOffsets[state], _arcOffsets[state + 1] - _arcOffsets[state]);

    /// <summary>Reads a binary <c>vector</c>/<c>standard</c> FST file.</summary>
    public static StdVectorFst Read(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using FileStream stream = new(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16);
        try
        {
            return Read(stream);
        }
        catch (InvalidDataException error)
        {
            throw new InvalidDataException($"{path}: {error.Message}", error);
        }
    }

    /// <summary>Reads a binary <c>vector</c>/<c>standard</c> FST from <paramref name="stream"/>.</summary>
    public static StdVectorFst Read(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using BinaryReader reader = new(stream, Encoding.UTF8, leaveOpen: true);
        try
        {
            if (reader.ReadInt32() != Magic) throw new InvalidDataException("not an OpenFst binary file");
            string fstType = ReadString(reader);
            string arcType = ReadString(reader);
            int version = reader.ReadInt32();
            int flags = reader.ReadInt32();
            reader.ReadUInt64();
            long start = reader.ReadInt64();
            long stateCount = reader.ReadInt64();
            reader.ReadInt64();
            if (fstType != "vector") throw new InvalidDataException($"FST type '{fstType}' is not supported");
            if (arcType != "standard") throw new InvalidDataException($"arc type '{arcType}' is not supported");
            if (version < MinVectorVersion) throw new InvalidDataException($"vector FST version {version} is too old");
            if ((flags & (HasInputSymbols | HasOutputSymbols)) != 0)
                throw new InvalidDataException("FSTs with embedded symbol tables are not supported");
            if (stateCount < 0 || stateCount > int.MaxValue - 1 || start < -1 || start >= stateCount)
                throw new InvalidDataException($"invalid header: start {start}, {stateCount} states");

            int states = (int)stateCount;
            float[] finals = new float[states];
            int[] offsets = new int[states + 1];
            List<StdArc> arcs = new(states * 2);
            for (int s = 0; s < states; s++)
            {
                finals[s] = reader.ReadSingle();
                long count = reader.ReadInt64();
                if (count < 0 || count > int.MaxValue - arcs.Count)
                    throw new InvalidDataException($"state {s} declares {count} arcs");
                offsets[s] = arcs.Count;
                for (long a = 0; a < count; a++)
                {
                    StdArc arc = new(reader.ReadInt32(), reader.ReadInt32(), reader.ReadSingle(), reader.ReadInt32());
                    if (arc.NextState < 0 || arc.NextState >= states)
                        throw new InvalidDataException($"arc from state {s} targets missing state {arc.NextState}");
                    arcs.Add(arc);
                }
            }
            offsets[states] = arcs.Count;
            return new StdVectorFst((int)start, finals, offsets, arcs.ToArray());
        }
        catch (EndOfStreamException error)
        {
            throw new InvalidDataException("FST file is truncated", error);
        }
    }

    private static string ReadString(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        if (length < 0 || length > 256) throw new InvalidDataException($"invalid header string length {length}");
        return Encoding.ASCII.GetString(reader.ReadBytes(length));
    }

    private bool ComputeILabelSorted()
    {
        for (int s = 0; s < StateCount; s++)
        {
            for (int a = _arcOffsets[s] + 1; a < _arcOffsets[s + 1]; a++)
                if (_arcs[a - 1].ILabel > _arcs[a].ILabel) return false;
        }
        return true;
    }

    // Iterative three-colour DFS over every state; a grey target is a back edge.
    private bool ComputeAcyclic()
    {
        byte[] color = new byte[StateCount];
        Stack<(int State, int Arc)> stack = new();
        for (int root = 0; root < StateCount; root++)
        {
            if (color[root] != 0) continue;
            color[root] = 1;
            stack.Push((root, _arcOffsets[root]));
            while (stack.Count > 0)
            {
                (int s, int a) = stack.Pop();
                if (a == _arcOffsets[s + 1])
                {
                    color[s] = 2;
                    continue;
                }
                stack.Push((s, a + 1));
                int t = _arcs[a].NextState;
                if (color[t] == 1) return false;
                if (color[t] != 0) continue;
                color[t] = 1;
                stack.Push((t, _arcOffsets[t]));
            }
        }
        return true;
    }
}
