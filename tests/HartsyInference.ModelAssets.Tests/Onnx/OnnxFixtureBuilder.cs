using System.Text;

namespace HartsyInference.ModelAssets.Tests.Onnx;

/// <summary>Writes a minimal ONNX <c>ModelProto</c> (graph nodes with int/float attributes, raw-data F32 initializers) so reader and converter tests need no file.</summary>
internal sealed class OnnxFixtureBuilder
{
    private readonly List<byte[]> _nodes = [];
    private readonly List<byte[]> _initializers = [];

    public void AddInitializer(string name, float[] data, params long[] dims)
    {
        List<byte> t = [];
        foreach (long d in dims) { WriteTag(t, 1, 0); WriteVarint(t, (ulong)d); }
        WriteTag(t, 2, 0); WriteVarint(t, 1);
        WriteString(t, 8, name);
        byte[] raw = new byte[data.Length * sizeof(float)];
        Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
        WriteBytes(t, 9, raw);
        _initializers.Add([.. t]);
    }

    public void AddNode(string opType, string name, string[] inputs, string[] outputs,
        Dictionary<string, long[]>? ints = null, Dictionary<string, float>? floats = null)
    {
        List<byte> n = [];
        foreach (string i in inputs) WriteString(n, 1, i);
        foreach (string o in outputs) WriteString(n, 2, o);
        WriteString(n, 3, name);
        WriteString(n, 4, opType);
        foreach (KeyValuePair<string, long[]> a in ints ?? [])
        {
            List<byte> attr = [];
            WriteString(attr, 1, a.Key);
            foreach (long v in a.Value) { WriteTag(attr, 8, 0); WriteVarint(attr, (ulong)v); }
            WriteBytes(n, 5, [.. attr]);
        }
        foreach (KeyValuePair<string, float> a in floats ?? [])
        {
            List<byte> attr = [];
            WriteString(attr, 1, a.Key);
            WriteTag(attr, 2, 5);
            attr.AddRange(BitConverter.GetBytes(a.Value));
            WriteBytes(n, 5, [.. attr]);
        }
        _nodes.Add([.. n]);
    }

    public byte[] Build()
    {
        List<byte> graph = [];
        foreach (byte[] n in _nodes) WriteBytes(graph, 1, n);
        foreach (byte[] i in _initializers) WriteBytes(graph, 5, i);
        List<byte> model = [];
        WriteBytes(model, 7, [.. graph]);
        return [.. model];
    }

    private static void WriteTag(List<byte> b, int field, int wire) => WriteVarint(b, (ulong)((field << 3) | wire));

    private static void WriteVarint(List<byte> b, ulong v)
    {
        while (v >= 0x80) { b.Add((byte)(v | 0x80)); v >>= 7; }
        b.Add((byte)v);
    }

    private static void WriteBytes(List<byte> b, int field, byte[] payload)
    {
        WriteTag(b, field, 2);
        WriteVarint(b, (ulong)payload.Length);
        b.AddRange(payload);
    }

    private static void WriteString(List<byte> b, int field, string s) => WriteBytes(b, field, Encoding.UTF8.GetBytes(s));
}
