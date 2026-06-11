using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;

namespace SharpInference.Tests.Common;

/// <summary>Synthetic weights for the ACE-Step v1 stack (DiT incl. lyric Conformer, Music-DCAE decoder, ADaMoS
/// vocoder) at tiny dims for CPU structural tests.</summary>
public static unsafe class AceStepSyntheticWeights
{
    private static int s_seed = 21000;

    /// <summary>Tiny DiT config used across the ACE-Step tests (dim 16, 2 heads × 8, lyric hidden 16).</summary>
    public static AceStepConfig TinyConfig => new()
    {
        NumHeads = 2, HeadDim = 8, NumLayers = 2, MlpRatio = 2.0f,
        PatchSize = (4, 1), LatentHeight = 4, PatchEmbedHidden = 8, FreqDim = 16,
        TextDim = 12, SpeakerDim = 8, LyricHiddenDim = 16, LyricVocabSize = 50, LyricEncoderLayers = 2,
        NumInferenceSteps = 2,
    };

    public static Dictionary<string, Tensor> BuildDit(AceStepConfig c)
    {
        int dim = c.InnerDim, hid = c.PatchEmbedHidden, lh = c.LyricHiddenDim;
        Dictionary<string, Tensor> w = new()
        {
            ["proj_in.early_conv_layers.0.weight"] = R([hid, c.InChannels, c.PatchSize.H, c.PatchSize.W]),
            ["proj_in.early_conv_layers.0.bias"] = R([hid]),
            ["proj_in.early_conv_layers.1.weight"] = R([hid]),
            ["proj_in.early_conv_layers.1.bias"] = R([hid]),
            ["proj_in.early_conv_layers.2.weight"] = R([dim, hid, 1, 1]),
            ["proj_in.early_conv_layers.2.bias"] = R([dim]),
            ["timestep_embedder.linear_1.weight"] = R([dim, c.FreqDim]),
            ["timestep_embedder.linear_1.bias"] = R([dim]),
            ["timestep_embedder.linear_2.weight"] = R([dim, dim]),
            ["timestep_embedder.linear_2.bias"] = R([dim]),
            ["t_block.1.weight"] = R([6 * dim, dim]),
            ["t_block.1.bias"] = R([6 * dim]),
            ["speaker_embedder.weight"] = R([dim, c.SpeakerDim]),
            ["speaker_embedder.bias"] = R([dim]),
            ["genre_embedder.weight"] = R([dim, c.TextDim]),
            ["genre_embedder.bias"] = R([dim]),
            ["lyric_embs.weight"] = R([c.LyricVocabSize, lh]),
            ["lyric_proj.weight"] = R([dim, lh]),
            ["lyric_proj.bias"] = R([dim]),
            ["final_layer.scale_shift_table"] = R([2, dim]),
            ["final_layer.linear.weight"] = R([c.LatentHeight * c.InChannels, dim]),
            ["final_layer.linear.bias"] = R([c.LatentHeight * c.InChannels]),
        };

        // Lyric Conformer (no macaron in the tiny build — presence-driven).
        w["lyric_encoder.embed.out.0.weight"] = R([lh, lh]); w["lyric_encoder.embed.out.0.bias"] = R([lh]);
        w["lyric_encoder.embed.out.1.weight"] = R([lh]); w["lyric_encoder.embed.out.1.bias"] = R([lh]);
        for (int i = 0; i < c.LyricEncoderLayers; i++)
        {
            string p = $"lyric_encoder.encoders.{i}";
            foreach (string norm in new[] { "norm_mha", "norm_ff", "norm_conv", "norm_final" })
            {
                w[$"{p}.{norm}.weight"] = R([lh]);
                w[$"{p}.{norm}.bias"] = R([lh]);
            }
            foreach (string lin in new[] { "linear_q", "linear_k", "linear_v", "linear_out" })
            {
                w[$"{p}.self_attn.{lin}.weight"] = R([lh, lh]);
                w[$"{p}.self_attn.{lin}.bias"] = R([lh]);
            }
            w[$"{p}.self_attn.linear_pos.weight"] = R([lh, lh]);
            w[$"{p}.self_attn.pos_bias_u"] = R([lh]);
            w[$"{p}.self_attn.pos_bias_v"] = R([lh]);
            w[$"{p}.feed_forward.w_1.weight"] = R([2 * lh, lh]); w[$"{p}.feed_forward.w_1.bias"] = R([2 * lh]);
            w[$"{p}.feed_forward.w_2.weight"] = R([lh, 2 * lh]); w[$"{p}.feed_forward.w_2.bias"] = R([lh]);
            w[$"{p}.conv_module.pointwise_conv1.weight"] = R([2 * lh, lh, 1]); w[$"{p}.conv_module.pointwise_conv1.bias"] = R([2 * lh]);
            w[$"{p}.conv_module.depthwise_conv.weight"] = R([lh, 1, 5]); w[$"{p}.conv_module.depthwise_conv.bias"] = R([lh]);
            w[$"{p}.conv_module.pointwise_conv2.weight"] = R([lh, lh, 1]); w[$"{p}.conv_module.pointwise_conv2.bias"] = R([lh]);
            w[$"{p}.conv_module.norm.weight"] = R([lh]); w[$"{p}.conv_module.norm.bias"] = R([lh]);
            w[$"{p}.conv_module.norm.running_mean"] = R([lh]); w[$"{p}.conv_module.norm.running_var"] = Ones([lh]);
        }
        w["lyric_encoder.after_norm.weight"] = R([lh]); w["lyric_encoder.after_norm.bias"] = R([lh]);

        int ffHidden = (int)(dim * c.MlpRatio);
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"transformer_blocks.{i}";
            w[$"{p}.scale_shift_table"] = R([6, dim]);
            foreach (string attn in new[] { "attn", "cross_attn" })
            {
                w[$"{p}.{attn}.to_q.weight"] = R([dim, dim]); w[$"{p}.{attn}.to_q.bias"] = R([dim]);
                w[$"{p}.{attn}.to_k.weight"] = R([dim, dim]); w[$"{p}.{attn}.to_k.bias"] = R([dim]);
                w[$"{p}.{attn}.to_v.weight"] = R([dim, dim]); w[$"{p}.{attn}.to_v.bias"] = R([dim]);
                w[$"{p}.{attn}.to_out.0.weight"] = R([dim, dim]);
                w[$"{p}.{attn}.to_out.0.bias"] = R([dim]);
            }
            w[$"{p}.ff.inverted_conv.conv.weight"] = R([2 * ffHidden, dim, 1]);
            w[$"{p}.ff.inverted_conv.conv.bias"] = R([2 * ffHidden]);
            w[$"{p}.ff.depth_conv.conv.weight"] = R([2 * ffHidden, 1, 3]);
            w[$"{p}.ff.depth_conv.conv.bias"] = R([2 * ffHidden]);
            w[$"{p}.ff.point_conv.conv.weight"] = R([dim, ffHidden, 1]);
        }
        return w;
    }

    /// <summary>Tiny DCAE decoder: blockOut [4,8,16,32], 1 layer per stage, attention (no multiscale) in the deepest.</summary>
    public static Dictionary<string, Tensor> BuildDcaeDecoder(int latentChannels = 8, int[]? blockOut = null, int glumbHidden = 8)
    {
        int[] dims = blockOut ?? [4, 8, 16, 32];
        Dictionary<string, Tensor> w = new()
        {
            ["decoder.conv_in.weight"] = R([dims[^1], latentChannels, 3, 3]),
            ["decoder.conv_in.bias"] = R([dims[^1]]),
            ["decoder.norm_out.weight"] = R([dims[0]]),
            ["decoder.norm_out.bias"] = R([dims[0]]),
            ["decoder.conv_out.weight"] = R([2, dims[0], 3, 3]),
            ["decoder.conv_out.bias"] = R([2]),
        };
        for (int i = 0; i < dims.Length; i++)
        {
            int child = 0;
            if (i < dims.Length - 1)
            {
                w[$"decoder.up_blocks.{i}.{child}.conv.weight"] = R([dims[i], dims[i + 1], 3, 3]);
                w[$"decoder.up_blocks.{i}.{child}.conv.bias"] = R([dims[i]]);
                child++;
            }
            string p = $"decoder.up_blocks.{i}.{child}";
            if (i == dims.Length - 1)
            {
                int d = dims[i];
                w[$"{p}.attn.to_q.weight"] = R([d, d]);
                w[$"{p}.attn.to_k.weight"] = R([d, d]);
                w[$"{p}.attn.to_v.weight"] = R([d, d]);
                w[$"{p}.attn.to_out.weight"] = R([d, d]);
                w[$"{p}.attn.norm_out.weight"] = R([d]);
                w[$"{p}.attn.norm_out.bias"] = R([d]);
                w[$"{p}.conv_out.conv_inverted.weight"] = R([2 * glumbHidden, d, 1, 1]);
                w[$"{p}.conv_out.conv_inverted.bias"] = R([2 * glumbHidden]);
                w[$"{p}.conv_out.conv_depth.weight"] = R([2 * glumbHidden, 1, 3, 3]);
                w[$"{p}.conv_out.conv_depth.bias"] = R([2 * glumbHidden]);
                w[$"{p}.conv_out.conv_point.weight"] = R([d, glumbHidden, 1, 1]);
                w[$"{p}.conv_out.norm.weight"] = R([d]);
            }
            else
            {
                w[$"{p}.conv1.weight"] = R([dims[i], dims[i], 3, 3]);
                w[$"{p}.conv1.bias"] = R([dims[i]]);
                w[$"{p}.conv2.weight"] = R([dims[i], dims[i], 3, 3]);
                w[$"{p}.norm.weight"] = R([dims[i]]);
            }
        }
        return w;
    }

    /// <summary>Tiny ADaMoS vocoder: mel bins configurable, 2 upsample stages of rate 2, 1 resblock kernel.</summary>
    public static Dictionary<string, Tensor> BuildVocoder(int melBins, int[]? dims = null)
    {
        int[] d = dims ?? [4, 6, 8, 8];
        Dictionary<string, Tensor> w = new()
        {
            ["backbone.channel_layers.0.0.weight"] = R([d[0], melBins, 7]),
            ["backbone.channel_layers.0.0.bias"] = R([d[0]]),
            ["backbone.channel_layers.0.1.weight"] = R([d[0]]),
            ["backbone.channel_layers.0.1.bias"] = R([d[0]]),
            ["backbone.norm.weight"] = R([d[^1]]),
            ["backbone.norm.bias"] = R([d[^1]]),
            ["head.conv_pre.weight"] = R([8, d[^1], 5]),
            ["head.conv_pre.bias"] = R([8]),
            ["head.ups.0.weight"] = R([8, 4, 4]),
            ["head.ups.0.bias"] = R([4]),
            ["head.ups.1.weight"] = R([4, 2, 4]),
            ["head.ups.1.bias"] = R([2]),
            ["head.conv_post.weight"] = R([1, 2, 5]),
            ["head.conv_post.bias"] = R([1]),
        };
        for (int i = 1; i < d.Length; i++)
        {
            w[$"backbone.channel_layers.{i}.0.weight"] = R([d[i - 1]]);
            w[$"backbone.channel_layers.{i}.0.bias"] = R([d[i - 1]]);
            w[$"backbone.channel_layers.{i}.1.weight"] = R([d[i], d[i - 1], 1]);
            w[$"backbone.channel_layers.{i}.1.bias"] = R([d[i]]);
        }
        for (int i = 0; i < d.Length; i++)
        {
            string p = $"backbone.stages.{i}.0";
            w[$"{p}.dwconv.weight"] = R([d[i], 1, 7]); w[$"{p}.dwconv.bias"] = R([d[i]]);
            w[$"{p}.norm.weight"] = R([d[i]]); w[$"{p}.norm.bias"] = R([d[i]]);
            w[$"{p}.pwconv1.weight"] = R([2 * d[i], d[i]]); w[$"{p}.pwconv1.bias"] = R([2 * d[i]]);
            w[$"{p}.pwconv2.weight"] = R([d[i], 2 * d[i]]); w[$"{p}.pwconv2.bias"] = R([d[i]]);
            w[$"{p}.gamma"] = Ones([d[i]]);
        }
        int[] resChannels = [4, 2];
        for (int i = 0; i < resChannels.Length; i++)
            for (int j = 0; j < 3; j++)
            {
                w[$"head.resblocks.{i}.convs1.{j}.weight"] = R([resChannels[i], resChannels[i], 3]);
                w[$"head.resblocks.{i}.convs1.{j}.bias"] = R([resChannels[i]]);
                w[$"head.resblocks.{i}.convs2.{j}.weight"] = R([resChannels[i], resChannels[i], 3]);
                w[$"head.resblocks.{i}.convs2.{j}.bias"] = R([resChannels[i]]);
            }
        return w;
    }

    private static Tensor Ones(int[] dims)
    {
        Tensor t = R(dims);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = 1f;
        return t;
    }

    private static Tensor R(int[] dims)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(s_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }
}
