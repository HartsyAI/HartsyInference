using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Seq2Seq;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using static HartsyInference.Tests.Common.Attention.AttentionTraceOps;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.LLM.Tests.Attention;

/// <summary>The op sequence of <see cref="T5Model"/>'s attention on the recording stub (no compute): every backend op in order, the attention
/// call's scale, F16 assertion and mask, the operand and output shapes, and which of those tensors are freed once the pass has returned. The pins
/// were recorded from the code before the site moved onto the shared layer and must not change. The self and cross masks are tensors, the cross
/// one all-zero rather than null.</summary>
public sealed class AttentionWaveE4TraceTests
{
    private const int D = 16, Heads = 2, Hd = 8, Inner = Heads * Hd, Inter = 24, Vocab = 20, Buckets = 8;

    private static T5Model Model()
    {
        Weights w = new(500);
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
        w.AddNorm("enc.blk.0.attn_norm.weight", D); Attn("enc.blk.0", "attn"); Ffn("enc.blk.0");
        w.AddNorm("dec.blk.0.attn_norm.weight", D); Attn("dec.blk.0", "attn");
        w.AddNorm("dec.blk.0.cross_attn_norm.weight", D); Attn("dec.blk.0", "cross_attn"); Ffn("dec.blk.0");
        return T5Model.FromWeights(w, D, 1, Heads, Hd, Inter, Buckets, 0, 1e-6f);
    }

    private static void Pin(string site, string[] actual)
    {
        if (!AttentionTraceOps.TryRecord(site, actual)) Assert.Equal(AttentionWaveE4TracePins.All[site], actual);
    }

    [Fact]
    public void Encode_ProjectsPermutesAttendsWithTheBiasMaskAndMergesHeads() =>
        Pin("t5model_encode", Record(b => Model().Encode(b, [3, 7, 1, 12, 5, 9, 2, 14]).Dispose()));

    [Fact]
    public void DecodeLogits_AttendsSelfCausallyThenCrossWithTheAllZeroMaskTensor() =>
        Pin("t5model_decode", Record(b =>
        {
            T5Model model = Model();
            using Tensor enc = new(new TensorShape(1, 6, D), DType.F32);
            model.DecodeLogits(b, enc, 6, [0, 4, 11]);
        }));

    [Fact]
    public void DecodeLogits_RefusesAnEncoderOutputThatIsNotEncLenRows()
    {
        using Tensor enc = new(new TensorShape(1, 8, D), DType.F32);
        ArgumentException e = Assert.Throws<ArgumentException>(() => Model().DecodeLogits(new AttentionTraceBackend(), enc, 6, [0, 4, 11]));
        Assert.Equal("encLen", e.ParamName);
    }
}
