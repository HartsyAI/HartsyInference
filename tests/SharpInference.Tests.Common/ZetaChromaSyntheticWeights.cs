using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;

namespace SharpInference.Tests.Common;

/// <summary>Synthetic weights for Zeta-Chroma structural tests: a tiny full Z-Image S3-DiT backbone (single-file
/// naming, no <c>final_layer.*</c>) plus the <c>dec_net.*</c> SimpleMLPAdaLN pixel decoder head.</summary>
public static unsafe class ZetaChromaSyntheticWeights
{
    private static int _seed = 15000;

    /// <summary>Builds the complete weight dict for a tiny Zeta-Chroma model.</summary>
    /// <param name="c">Zeta config; backbone dims are read from <c>c.Backbone</c>.</param>
    /// <param name="tEmbMlpHidden">Hidden width of the t_embedder MLP (independent of AdaLNEmbedDim).</param>
    public static Dictionary<string, Tensor> Build(ZetaChromaConfig c, int tEmbMlpHidden = 32)
    {
        ZImageConfig b = c.Backbone;
        int hidden = b.HiddenSize;
        int adaLN = b.AdaLNEmbedDim;
        int patchDim = b.InChannels * b.PatchSize * b.PatchSize;
        int decC = c.DecoderHidden;
        int decIn = patchDim + c.DecoderMaxFreqs * c.DecoderMaxFreqs;

        Dictionary<string, Tensor> w = new()
        {
            ["t_embedder.mlp.0.weight"] = R([tEmbMlpHidden, adaLN]),
            ["t_embedder.mlp.0.bias"] = R([tEmbMlpHidden]),
            ["t_embedder.mlp.2.weight"] = R([adaLN, tEmbMlpHidden]),
            ["t_embedder.mlp.2.bias"] = R([adaLN]),
            ["cap_embedder.0.weight"] = R([b.CapFeatDim]),
            ["cap_embedder.1.weight"] = R([hidden, b.CapFeatDim]),
            ["cap_embedder.1.bias"] = R([hidden]),
            ["x_embedder.weight"] = R([hidden, patchDim]),
            ["x_embedder.bias"] = R([hidden]),
            ["cap_pad_token"] = R([1, hidden]),
            ["x_pad_token"] = R([1, hidden]),
            ["dec_net.cond_embed.weight"] = R([decC, hidden]),
            ["dec_net.cond_embed.bias"] = R([decC]),
            ["dec_net.input_embedder.embedder.0.weight"] = R([decC, decIn]),
            ["dec_net.input_embedder.embedder.0.bias"] = R([decC]),
            ["dec_net.final_layer.linear.weight"] = R([patchDim, decC]),
            ["dec_net.final_layer.linear.bias"] = R([patchDim]),
        };

        for (int i = 0; i < c.DecoderResBlocks; i++)
        {
            string p = $"dec_net.res_blocks.{i}";
            w[$"{p}.in_ln.weight"] = R([decC]);
            w[$"{p}.in_ln.bias"] = R([decC]);
            w[$"{p}.mlp.0.weight"] = R([decC, decC]);
            w[$"{p}.mlp.0.bias"] = R([decC]);
            w[$"{p}.mlp.2.weight"] = R([decC, decC]);
            w[$"{p}.mlp.2.bias"] = R([decC]);
            w[$"{p}.adaLN_modulation.1.weight"] = R([3 * decC, decC]);
            w[$"{p}.adaLN_modulation.1.bias"] = R([3 * decC]);
        }

        for (int i = 0; i < b.NumRefinerLayers; i++)
        {
            AddMainBlock(w, $"noise_refiner.{i}", hidden, b.HeadDim, b.FfnDim, adaLN);
            AddContextBlock(w, $"context_refiner.{i}", hidden, b.HeadDim, b.FfnDim);
        }
        for (int i = 0; i < b.NumLayers; i++)
            AddMainBlock(w, $"layers.{i}", hidden, b.HeadDim, b.FfnDim, adaLN);

        return w;
    }

    private static void AddMainBlock(Dictionary<string, Tensor> w, string p, int hidden, int headDim, int ffn, int adaLN)
    {
        w[$"{p}.adaLN_modulation.0.weight"] = R([4 * hidden, adaLN]);
        w[$"{p}.adaLN_modulation.0.bias"] = R([4 * hidden]);
        AddContextBlock(w, p, hidden, headDim, ffn);
    }

    private static void AddContextBlock(Dictionary<string, Tensor> w, string p, int hidden, int headDim, int ffn)
    {
        w[$"{p}.attention.qkv.weight"] = R([3 * hidden, hidden]);
        w[$"{p}.attention.out.weight"] = R([hidden, hidden]);
        w[$"{p}.attention.q_norm.weight"] = R([headDim]);
        w[$"{p}.attention.k_norm.weight"] = R([headDim]);
        w[$"{p}.attention_norm1.weight"] = R([hidden]);
        w[$"{p}.attention_norm2.weight"] = R([hidden]);
        w[$"{p}.ffn_norm1.weight"] = R([hidden]);
        w[$"{p}.ffn_norm2.weight"] = R([hidden]);
        w[$"{p}.feed_forward.w1.weight"] = R([ffn, hidden]);
        w[$"{p}.feed_forward.w2.weight"] = R([hidden, ffn]);
        w[$"{p}.feed_forward.w3.weight"] = R([ffn, hidden]);
    }

    private static Tensor R(int[] dims)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }
}
