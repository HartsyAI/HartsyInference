using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Tests.Common;

/// <summary>Shared synthetic-weight generators for the Kandinsky 5 video stack: the
/// <see cref="Kandinsky5Transformer"/> (T2I or T2V visual-cond variants, diffusers key names) and the
/// HunyuanVideo VAE (encoder + decoder + quant convs). One source of truth for the Diffusion + Video test projects.</summary>
public static unsafe class Kandinsky5SyntheticWeights
{
    private static int _seed = 7000;

    /// <summary>Tiny T2V config (visual_cond, 33-ch-style packing at 4 latent channels → 9-ch input).</summary>
    public static Kandinsky5Config TinyVideoConfig => new()
    {
        InVisualDim = 4,
        OutVisualDim = 4,
        TimeDim = 16,
        PatchSize = (1, 2, 2),
        ModelDim = 32,
        FfDim = 64,
        NumTextBlocks = 1,
        NumVisualBlocks = 2,
        AxesDims = [4, 6, 6],
        InTextDim = 8,
        InTextDim2 = 12,
        VisualCond = true,
    };

    /// <summary>Tiny T2I config sharing dims with <see cref="TinyVideoConfig"/> but without visual cond.</summary>
    public static Kandinsky5Config TinyImageConfig => TinyVideoConfig with { VisualCond = false };

    /// <summary>Tiny HunyuanVideo VAE config (4-ch latent, same 8×/4× compression rules as the real model).</summary>
    public static HunyuanVideoVaeConfig TinyVaeConfig => new()
    {
        InChannels = 3,
        LatentChannels = 4,
        BlockOutChannels = [8, 8, 16, 16],
        LayersPerBlock = 1,
        NormGroups = 4,
        ScalingFactor = 0.476986f,
    };

    /// <summary>Builds a full diffusers-keyed transformer state dict for the given config.</summary>
    public static Dictionary<string, Tensor> BuildTransformer(Kandinsky5Config c)
    {
        int dim = c.ModelDim, timeDim = c.TimeDim, ff = c.FfDim, headDim = c.HeadDim;
        (int pT, int pH, int pW) = c.PatchSize;
        int patchIn = pT * pH * pW * c.VisualEmbedDim;
        int patchOut = pT * pH * pW * c.OutVisualDim;

        Dictionary<string, Tensor> w = new()
        {
            ["text_embeddings.in_layer.weight"] = R([dim, c.InTextDim]),
            ["text_embeddings.in_layer.bias"] = R([dim]),
            ["text_embeddings.norm.weight"] = R([dim]),
            ["text_embeddings.norm.bias"] = R([dim]),
            ["pooled_text_embeddings.in_layer.weight"] = R([timeDim, c.InTextDim2]),
            ["pooled_text_embeddings.in_layer.bias"] = R([timeDim]),
            ["pooled_text_embeddings.norm.weight"] = R([timeDim]),
            ["pooled_text_embeddings.norm.bias"] = R([timeDim]),
            ["time_embeddings.in_layer.weight"] = R([timeDim, dim]),
            ["time_embeddings.in_layer.bias"] = R([timeDim]),
            ["time_embeddings.out_layer.weight"] = R([timeDim, timeDim]),
            ["time_embeddings.out_layer.bias"] = R([timeDim]),
            ["visual_embeddings.in_layer.weight"] = R([dim, patchIn]),
            ["visual_embeddings.in_layer.bias"] = R([dim]),
            ["out_layer.modulation.out_layer.weight"] = R([2 * dim, timeDim]),
            ["out_layer.modulation.out_layer.bias"] = R([2 * dim]),
            ["out_layer.out_layer.weight"] = R([patchOut, dim]),
            ["out_layer.out_layer.bias"] = R([patchOut]),
        };

        for (int i = 0; i < c.NumTextBlocks; i++)
        {
            string p = $"text_transformer_blocks.{i}";
            w[$"{p}.text_modulation.out_layer.weight"] = R([6 * dim, timeDim]);
            w[$"{p}.text_modulation.out_layer.bias"] = R([6 * dim]);
            AddAttention(w, $"{p}.self_attention", dim, headDim);
            w[$"{p}.feed_forward.in_layer.weight"] = R([ff, dim]);
            w[$"{p}.feed_forward.out_layer.weight"] = R([dim, ff]);
        }

        for (int i = 0; i < c.NumVisualBlocks; i++)
        {
            string p = $"visual_transformer_blocks.{i}";
            w[$"{p}.visual_modulation.out_layer.weight"] = R([9 * dim, timeDim]);
            w[$"{p}.visual_modulation.out_layer.bias"] = R([9 * dim]);
            AddAttention(w, $"{p}.self_attention", dim, headDim);
            AddAttention(w, $"{p}.cross_attention", dim, headDim);
            w[$"{p}.feed_forward.in_layer.weight"] = R([ff, dim]);
            w[$"{p}.feed_forward.out_layer.weight"] = R([dim, ff]);
        }

        return w;
    }

    /// <summary>Builds a full diffusers-keyed HunyuanVideo VAE state dict (encoder + decoder + quant convs).</summary>
    public static Dictionary<string, Tensor> BuildVae(HunyuanVideoVaeConfig c)
    {
        Dictionary<string, Tensor> w = new();
        int[] ch = c.BlockOutChannels;
        int latent = c.LatentChannels;

        // ── Encoder ──
        AddConv(w, "encoder.conv_in", ch[0], c.InChannels, 3);
        int prev = ch[0];
        for (int i = 0; i < ch.Length; i++)
        {
            int outCh = ch[i];
            int cur = prev;
            for (int j = 0; j < c.LayersPerBlock; j++)
            {
                AddResnet(w, $"encoder.down_blocks.{i}.resnets.{j}", cur, outCh);
                cur = outCh;
            }
            if (c.StageHasSpatialResample(i) || c.StageHasTemporalResample(i))
                AddConv(w, $"encoder.down_blocks.{i}.downsamplers.0.conv", outCh, outCh, 3);
            prev = outCh;
        }
        AddMidBlock(w, "encoder.mid_block", ch[^1], c.MidBlockAttention);
        w["encoder.conv_norm_out.weight"] = R([ch[^1]]);
        w["encoder.conv_norm_out.bias"] = R([ch[^1]]);
        AddConv(w, "encoder.conv_out", 2 * latent, ch[^1], 3);
        AddConv1x1(w, "quant_conv", 2 * latent, 2 * latent);

        // ── Decoder ──
        AddConv1x1(w, "post_quant_conv", latent, latent);
        AddConv(w, "decoder.conv_in", ch[^1], latent, 3);
        AddMidBlock(w, "decoder.mid_block", ch[^1], c.MidBlockAttention);
        prev = ch[^1];
        for (int i = 0; i < ch.Length; i++)
        {
            int outCh = ch[ch.Length - 1 - i];
            int cur = prev;
            for (int j = 0; j < c.LayersPerBlock + 1; j++)
            {
                AddResnet(w, $"decoder.up_blocks.{i}.resnets.{j}", cur, outCh);
                cur = outCh;
            }
            if (c.StageHasSpatialResample(i) || c.StageHasTemporalResample(i))
                AddConv(w, $"decoder.up_blocks.{i}.upsamplers.0.conv", outCh, outCh, 3);
            prev = outCh;
        }
        w["decoder.conv_norm_out.weight"] = R([ch[0]]);
        w["decoder.conv_norm_out.bias"] = R([ch[0]]);
        AddConv(w, "decoder.conv_out", c.InChannels, ch[0], 3);

        return w;
    }

    private static void AddAttention(Dictionary<string, Tensor> w, string p, int dim, int headDim)
    {
        foreach (string lin in new[] { "to_query", "to_key", "to_value", "out_layer" })
        {
            w[$"{p}.{lin}.weight"] = R([dim, dim]);
            w[$"{p}.{lin}.bias"] = R([dim]);
        }
        w[$"{p}.query_norm.weight"] = R([headDim]);
        w[$"{p}.key_norm.weight"] = R([headDim]);
    }

    private static void AddResnet(Dictionary<string, Tensor> w, string p, int inCh, int outCh)
    {
        w[$"{p}.norm1.weight"] = R([inCh]);
        w[$"{p}.norm1.bias"] = R([inCh]);
        AddConv(w, $"{p}.conv1", outCh, inCh, 3);
        w[$"{p}.norm2.weight"] = R([outCh]);
        w[$"{p}.norm2.bias"] = R([outCh]);
        AddConv(w, $"{p}.conv2", outCh, outCh, 3);
        if (inCh != outCh)
            AddConv1x1(w, $"{p}.conv_shortcut.conv", outCh, inCh);
    }

    private static void AddMidBlock(Dictionary<string, Tensor> w, string p, int channels, bool attention)
    {
        AddResnet(w, $"{p}.resnets.0", channels, channels);
        AddResnet(w, $"{p}.resnets.1", channels, channels);
        if (!attention) return;
        w[$"{p}.attentions.0.group_norm.weight"] = R([channels]);
        w[$"{p}.attentions.0.group_norm.bias"] = R([channels]);
        foreach (string lin in new[] { "to_q", "to_k", "to_v", "to_out.0" })
        {
            w[$"{p}.attentions.0.{lin}.weight"] = R([channels, channels]);
            w[$"{p}.attentions.0.{lin}.bias"] = R([channels]);
        }
    }

    private static void AddConv(Dictionary<string, Tensor> w, string baseKey, int outCh, int inCh, int k)
    {
        w[$"{baseKey}.conv.weight"] = R([outCh, inCh, k, k, k]);
        w[$"{baseKey}.conv.bias"] = R([outCh]);
    }

    private static void AddConv1x1(Dictionary<string, Tensor> w, string baseKey, int outCh, int inCh)
    {
        w[$"{baseKey}.weight"] = R([outCh, inCh, 1, 1, 1]);
        w[$"{baseKey}.bias"] = R([outCh]);
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
