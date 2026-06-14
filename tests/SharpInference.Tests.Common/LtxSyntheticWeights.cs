using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;

namespace SharpInference.Tests.Common;

/// <summary>Shared synthetic-weight generators for LTX-Video structural tests (DiT transformer + base VAE decoder). One source of truth so the Diffusion and Video test projects don't duplicate the weight-dict layout. Small random values — these gate wiring/shape/finiteness, not numerics.</summary>
public static unsafe class LtxSyntheticWeights
{
    private static int _seed = 4000;

    /// <summary>Builds a full <see cref="LtxVideoTransformer"/> weight dict for the given (tiny) config.</summary>
    public static Dictionary<string, Tensor> BuildTransformer(LtxVideoConfig c)
    {
        int dim = c.InnerDim, ff = 4 * dim;
        Dictionary<string, Tensor> w = new()
        {
            ["proj_in.weight"] = R([dim, c.InChannels]), ["proj_in.bias"] = R([dim]),
            ["proj_out.weight"] = R([c.OutChannels, dim]), ["proj_out.bias"] = R([c.OutChannels]),
            ["scale_shift_table"] = R([2, dim]),
            ["time_embed.emb.timestep_embedder.linear_1.weight"] = R([dim, 256]), ["time_embed.emb.timestep_embedder.linear_1.bias"] = R([dim]),
            ["time_embed.emb.timestep_embedder.linear_2.weight"] = R([dim, dim]), ["time_embed.emb.timestep_embedder.linear_2.bias"] = R([dim]),
            ["time_embed.linear.weight"] = R([6 * dim, dim]), ["time_embed.linear.bias"] = R([6 * dim]),
            ["caption_projection.linear_1.weight"] = R([dim, c.CaptionChannels]), ["caption_projection.linear_1.bias"] = R([dim]),
            ["caption_projection.linear_2.weight"] = R([dim, dim]), ["caption_projection.linear_2.bias"] = R([dim]),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"transformer_blocks.{i}";
            w[$"{p}.scale_shift_table"] = R([6, dim]);
            foreach (string a in new[] { "attn1", "attn2" })
            {
                w[$"{p}.{a}.to_q.weight"] = R([dim, dim]); w[$"{p}.{a}.to_q.bias"] = R([dim]);
                w[$"{p}.{a}.to_k.weight"] = R([dim, dim]); w[$"{p}.{a}.to_k.bias"] = R([dim]);
                w[$"{p}.{a}.to_v.weight"] = R([dim, dim]); w[$"{p}.{a}.to_v.bias"] = R([dim]);
                w[$"{p}.{a}.to_out.0.weight"] = R([dim, dim]); w[$"{p}.{a}.to_out.0.bias"] = R([dim]);
                w[$"{p}.{a}.norm_q.weight"] = R([dim]); w[$"{p}.{a}.norm_k.weight"] = R([dim]);
            }
            w[$"{p}.ff.net.0.proj.weight"] = R([ff, dim]); w[$"{p}.ff.net.0.proj.bias"] = R([ff]);
            w[$"{p}.ff.net.2.weight"] = R([dim, ff]); w[$"{p}.ff.net.2.bias"] = R([dim]);
        }
        return w;
    }

    /// <summary>Builds a full base <see cref="SharpInference.Diffusion.Models.Vae.LtxVideoVaeDecoder"/> weight dict (no timestep conditioning).</summary>
    public static Dictionary<string, Tensor> BuildVaeDecoder(int latent, int outCh, int[] blockOut, bool[] scaling, int[] layers, int patch)
    {
        int[] bo = Rev(blockOut);
        bool[] sc = Rev(scaling);
        int[] ly = Rev(layers);
        Dictionary<string, Tensor> w = new();
        int output = bo[0];

        AddConv(w, "decoder.conv_in", output, latent, 3);
        for (int j = 0; j < ly[0]; j++) AddResnet(w, $"decoder.mid_block.resnets.{j}", output, output);

        for (int i = 0; i < bo.Length; i++)
        {
            int inC = output, outC = bo[i];
            string p = $"decoder.up_blocks.{i}";
            if (inC != outC) AddResnet(w, $"{p}.conv_in", inC, outC);
            if (sc[i]) AddConv(w, $"{p}.upsamplers.0.conv", outC * 8, outC, 3);
            for (int j = 0; j < ly[i + 1]; j++) AddResnet(w, $"{p}.resnets.{j}", outC, outC);
            output = outC;
        }
        AddConv(w, "decoder.conv_out", outCh * patch * patch, output, 3);
        return w;
    }

    private static void AddResnet(Dictionary<string, Tensor> w, string p, int inC, int outC)
    {
        AddConv(w, $"{p}.conv1", outC, inC, 3);
        AddConv(w, $"{p}.conv2", outC, outC, 3);
        if (inC != outC)
        {
            w[$"{p}.norm3.weight"] = R([inC]); w[$"{p}.norm3.bias"] = R([inC]);
            AddConv(w, $"{p}.conv_shortcut", outC, inC, 1);
        }
    }

    private static void AddConv(Dictionary<string, Tensor> w, string p, int outC, int inC, int k)
    {
        w[$"{p}.conv.weight"] = R([outC, inC, k, k, k]);
        w[$"{p}.conv.bias"] = R([outC]);
    }

    private static int[] Rev(int[] a) { int[] r = (int[])a.Clone(); Array.Reverse(r); return r; }
    private static bool[] Rev(bool[] a) { bool[] r = (bool[])a.Clone(); Array.Reverse(r); return r; }

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
