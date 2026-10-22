using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Seq2Seq;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.LLM.Tests.Attention;

/// <summary>Forward passes of <see cref="T5Model"/> (the GGUF T5 encoder-decoder, which has no consumer or KV cache) on tiny pseudo-random weights
/// built through its internal weight-dictionary seam, each reduced to SHA-256 digests of its output bits. Everything runs on the CPU backend, so the
/// same cases produce the golden fixture from the pre-migration source and must reproduce it after the attention refactor.</summary>
internal static unsafe class AttentionWaveEGoldenCases
{
    private const int D = 16, Layers = 2, Heads = 2, Hd = 8, Inner = Heads * Hd, Inter = 24, Vocab = 20, Buckets = 8;

    private static readonly Dictionary<string, Func<CpuBackend, string>> Cases = new(StringComparer.Ordinal)
    {
        ["t5model_encode"] = Encode,
        ["t5model_decode"] = Decode,
    };

    /// <summary>Every case, in fixture order.</summary>
    public static IReadOnlyList<string> Names { get; } = [.. Cases.Keys];

    /// <summary>Runs <paramref name="name"/> and returns its digests as space-separated <c>label=hex</c> pairs.</summary>
    public static string Run(string name)
    {
        using CpuBackend cpu = new();
        return Cases.TryGetValue(name, out Func<CpuBackend, string>? run)
            ? run(cpu)
            : throw new ArgumentException($"Unknown attention golden case '{name}'.", nameof(name));
    }

    private static Weights ModelWeights(uint salt)
    {
        Weights w = new(salt);
        w.Add("token_embd.weight", Vocab, D); w.Add("output.weight", Vocab, D);
        w.Add("enc.blk.0.attn_rel_b.weight", Buckets, Heads); w.Add("dec.blk.0.attn_rel_b.weight", Buckets, Heads);
        w.AddNorm("enc.output_norm.weight", D); w.AddNorm("dec.output_norm.weight", D);
        void Attn(string p, string name)
        {
            foreach (string n in new[] { "q", "k", "v" }) w.Add($"{p}.{name}_{n}.weight", Inner, D);
            w.Add($"{p}.{name}_o.weight", D, Inner);
        }
        void Ffn(string p)
        {
            w.AddNorm($"{p}.ffn_norm.weight", D);
            w.Add($"{p}.ffn_gate.weight", Inter, D); w.Add($"{p}.ffn_up.weight", Inter, D); w.Add($"{p}.ffn_down.weight", D, Inter);
        }
        for (int i = 0; i < Layers; i++)
        {
            string e = $"enc.blk.{i}", d = $"dec.blk.{i}";
            w.AddNorm($"{e}.attn_norm.weight", D); Attn(e, "attn"); Ffn(e);
            w.AddNorm($"{d}.attn_norm.weight", D); Attn(d, "attn");
            w.AddNorm($"{d}.cross_attn_norm.weight", D); Attn(d, "cross_attn"); Ffn(d);
        }
        return w;
    }

    private static string Encode(CpuBackend cpu)
    {
        Owner o = new();
        using T5Model model = T5Model.FromWeights(ModelWeights(500), D, Layers, Heads, Hd, Inter, Buckets, 0, 1e-6f);
        Tensor enc = o.Own(model.Encode(cpu, [3, 7, 1, 12, 5, 9, 2, 14]));
        return Finish(o, ("enc", enc));
    }

    private static string Decode(CpuBackend cpu)
    {
        Owner o = new();
        using T5Model model = T5Model.FromWeights(ModelWeights(501), D, Layers, Heads, Hd, Inter, Buckets, 0, 1e-6f);
        Tensor enc = o.Own(model.Encode(cpu, [3, 7, 1, 12, 5, 9]));
        // Each step re-runs the whole decoder prefix (no KV cache): causal self-attention plus the all-zero cross mask.
        List<string> digests = [];
        int[] decoded = [0, 4, 11, 6];
        for (int n = 1; n <= decoded.Length; n++)
            digests.Add(Digest($"step{n}", model.DecodeLogits(cpu, enc, 6, decoded.AsSpan(0, n).ToArray())));
        Finish(o);
        return string.Join(' ', digests);
    }
}
