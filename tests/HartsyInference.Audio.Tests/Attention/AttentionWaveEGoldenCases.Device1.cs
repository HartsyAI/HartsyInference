using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.F5Tts;
using HartsyInference.Audio.Models.IndexTts2;
using HartsyInference.Audio.Models.Kokoro;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Audio.Tests.Attention;

internal static unsafe partial class AttentionWaveEGoldenCases
{
    private static void RegisterDevice1(Dictionary<string, Func<CpuBackend, string>> d)
    {
        d["f5_block"] = F5Block;
        d["indextts2_dit_block"] = IndexTts2Block;
        d["kokoro_albert"] = KokoroAlbert;
        d["cosyvoice_matcha_block"] = MatchaBlock;
    }

    /// <summary>Rope table <c>[maxPos, headDim]</c> in the pair-repeated layout the WanRopeInterleaved users build (<c>c[2i] == c[2i+1]</c>).</summary>
    private static (Tensor Cos, Tensor Sin) PairRepeatedRope(Owner o, int headDim, int maxPos)
    {
        (float[] cos, float[] sin) = HartsyInference.Audio.Models.Moonshine.RotaryEmbedding.GetTables(headDim, 10_000f, maxPos);
        int half = headDim / 2;
        Tensor c = o.Own(new Tensor(new TensorShape(maxPos, headDim), DType.F32)), s = o.Own(new Tensor(new TensorShape(maxPos, headDim), DType.F32));
        float* cp = (float*)c.DataPointer, sp = (float*)s.DataPointer;
        for (int p = 0; p < maxPos; p++)
            for (int i = 0; i < half; i++)
            {
                cp[p * headDim + 2 * i] = cp[p * headDim + 2 * i + 1] = cos[p * half + i];
                sp[p * headDim + 2 * i] = sp[p * headDim + 2 * i + 1] = sin[p * half + i];
            }
        return (c, s);
    }

    private static string F5Block(CpuBackend cpu)
    {
        const int dim = 32, heads = 2, ff = 64, t = 9;
        Owner o = new();
        F5TtsConfig cfg = new() { Dim = dim, Heads = heads, FfMult = 2 };
        Weights w = new(81);
        w.Linear("b.attn_norm.linear", 6 * dim, dim, true);
        foreach (string n in new[] { "to_q", "to_k", "to_v" }) w.Linear($"b.attn.{n}", dim, dim, true);
        w.Linear("b.attn.to_out.0", dim, dim, true);
        w.Linear("b.ff.ff.0.0", ff, dim, true); w.Linear("b.ff.ff.2", dim, ff, true);
        F5DitBlock block = new(cfg);
        block.LoadWeights(w, "b");
        (Tensor cos, Tensor sin) = PairRepeatedRope(o, dim / heads, 32);
        Tensor x = Seq(o, 811, 1f, 1, t, dim), time = Seq(o, 812, 1f, 1, 1, dim);
        Tensor y = o.Own(block.Forward(cpu, x, time, t, cos, sin));
        return Finish(o, ("out", y));
    }

    private static string IndexTts2Block(CpuBackend cpu)
    {
        const int hidden = 32, heads = 2, ffn = 48, t = 9;
        Owner o = new();
        Weights w = new(82);
        foreach (string n in new[] { "attention_norm", "ffn_norm" })
        {
            w.AddNorm($"b.{n}.norm.weight", hidden); w.Linear($"b.{n}.project_layer", 2 * hidden, hidden, true);
        }
        w.Add("b.attention.wqkv.weight", 3 * hidden, hidden); w.Add("b.attention.wo.weight", hidden, hidden);
        w.Add("b.feed_forward.w1.weight", ffn, hidden); w.Add("b.feed_forward.w3.weight", ffn, hidden); w.Add("b.feed_forward.w2.weight", hidden, ffn);
        IndexTts2DitBlock block = new(hidden, heads, ffn);
        block.LoadWeights(w, "b");
        (Tensor cos, Tensor sin) = PairRepeatedRope(o, hidden / heads, 32);
        Tensor x = Seq(o, 821, 1f, 1, t, hidden), emb = Seq(o, 822, 1f, 1, hidden);
        Tensor y = o.Own(block.Forward(cpu, x, emb, t, cos, sin, null));
        return Finish(o, ("out", y));
    }

    private static string KokoroAlbert(CpuBackend cpu)
    {
        const int hidden = 24, heads = 2, inter = 32, embSize = 8, layers = 2, vocab = 16;
        Owner o = new();
        KokoroConfig cfg = new()
        {
            PlBert = new KokoroConfig.PlBertConfig
            {
                HiddenSize = hidden, NumAttentionHeads = heads, IntermediateSize = inter, MaxPositionEmbeddings = 16,
                NumHiddenLayers = layers, EmbeddingSize = embSize,
            },
        };
        Weights w = new(83);
        w.Add("bert.embeddings.word_embeddings.weight", vocab, embSize);
        w.Add("bert.embeddings.position_embeddings.weight", 16, embSize);
        w.Add("bert.embeddings.token_type_embeddings.weight", 2, embSize);
        w.AddNorm("bert.embeddings.LayerNorm.weight", embSize); w.AddBias("bert.embeddings.LayerNorm.bias", embSize);
        w.Linear("bert.encoder.embedding_hidden_mapping_in", hidden, embSize, true);
        const string p = "bert.encoder.albert_layer_groups.0.albert_layers.0";
        foreach (string n in new[] { "query", "key", "value", "dense" }) w.Linear($"{p}.attention.{n}", hidden, hidden, true);
        w.AddNorm($"{p}.attention.LayerNorm.weight", hidden); w.AddBias($"{p}.attention.LayerNorm.bias", hidden);
        w.Linear($"{p}.ffn", inter, hidden, true); w.Linear($"{p}.ffn_output", hidden, inter, true);
        w.AddNorm($"{p}.full_layer_layer_norm.weight", hidden); w.AddBias($"{p}.full_layer_layer_norm.bias", hidden);
        w.Linear("bert_encoder", 512, hidden, true);
        KokoroPlBert bert = new(cfg);
        bert.LoadWeights(w);
        Tensor y = o.Own(bert.Forward(cpu, new[] { 3, 7, 1, 12, 5, 9, 2 }));
        return Finish(o, ("out", y));
    }

    private static string MatchaBlock(CpuBackend cpu)
    {
        // inner width 32 against model width 16 (the real block is 512 inner against 256), with and without the chunk-causal mask.
        const int ch = 16, heads = 4, hd = 8, ff = 40, t = 10;
        Owner o = new();
        Weights w = new(84);
        w.AddNorm("m.norm1.weight", ch); w.AddBias("m.norm1.bias", ch);
        w.Add("m.attn1.to_q.weight", heads * hd, ch); w.Add("m.attn1.to_k.weight", heads * hd, ch); w.Add("m.attn1.to_v.weight", heads * hd, ch);
        w.Linear("m.attn1.to_out.0", ch, heads * hd, true);
        w.AddNorm("m.norm3.weight", ch); w.AddBias("m.norm3.bias", ch);
        w.Linear("m.ff.net.0.proj", ff, ch, true); w.Linear("m.ff.net.2", ch, ff, true);
        MatchaTransformerBlock block = new(ch, heads, hd);
        block.LoadWeights(w, "m");
        Tensor x = Seq(o, 841, 1f, 1, t, ch);
        Tensor plain = o.Own(block.Forward(cpu, x, t));
        Tensor mask = o.Own(MaskBuilder.BuildChunkCausalMask(t, 4));
        Tensor masked = o.Own(block.Forward(cpu, x, t, mask));
        return Finish(o, ("nomask", plain), ("chunkmask", masked));
    }
}
