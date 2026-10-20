using System.Text.Json;

namespace HartsyInference.GpuBenchmarks.Attention;

/// <summary>One row of the attention shape table, shared with the Python baseline through
/// <c>benchmarks/python-baseline/attention_shapes.json</c> so both engines time the same geometry.</summary>
/// <param name="Layout"><c>hm</c> head-major <c>[B,H,S,D]</c> or <c>tm</c> token-major <c>[B,S,H,D]</c>, the layout the model hands to attention.</param>
/// <param name="Mask"><c>none</c>, <c>bias-head</c> (additive F32 <c>[1,H,Sq,Skv]</c>, T5-style) or <c>bias-key</c> (additive F32 <c>[1,1,1,Skv]</c>).</param>
/// <param name="Entry"><c>sdpa</c> (ScaledDotProductAttention family) or <c>flash</c> (the LLM FlashAttention entry).</param>
public sealed record AttentionBenchShape(
    string Id, int B, int Hq, int Hkv, int Sq, int Skv, int D,
    string Layout, string Mask, string Entry, bool Causal, string Note)
{
    /// <summary>True when K/V carry fewer heads than Q.</summary>
    public bool Gqa => Hkv != Hq;

    /// <summary>Query position of row 0: decode and chunked prefill sit at the end of the cache.</summary>
    public int QOffset => Causal ? Skv - Sq : 0;

    /// <summary>Useful attention FLOPs: 4·B·Hq·Sq·Skv·D, halved for a square causal block.</summary>
    public double Flops => 4.0 * B * Hq * (double)Sq * Skv * D * (Causal && Sq == Skv ? 0.5 : 1.0);

    private static readonly JsonSerializerOptions Options = new() { PropertyNameCaseInsensitive = true };

    /// <summary>Loads the <c>shapes</c> array of an attention shape file.</summary>
    public static IReadOnlyList<AttentionBenchShape> Load(string path)
    {
        using FileStream stream = File.OpenRead(path);
        using JsonDocument doc = JsonDocument.Parse(stream);
        List<AttentionBenchShape> shapes = [];
        foreach (JsonElement e in doc.RootElement.GetProperty("shapes").EnumerateArray())
        {
            AttentionBenchShape s = e.Deserialize<AttentionBenchShape>(Options)
                ?? throw new InvalidDataException($"{path}: unreadable shape entry {e}");
            if (s.Hkv <= 0 || s.Hq % s.Hkv != 0)
                throw new InvalidDataException($"{path}: {s.Id} has Hq={s.Hq} not a multiple of Hkv={s.Hkv}");
            shapes.Add(s);
        }
        return shapes;
    }
}
