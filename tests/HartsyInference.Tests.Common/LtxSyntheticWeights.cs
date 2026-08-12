using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Tests.Common;

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

    /// <summary>Builds a full base <see cref="HartsyInference.Diffusion.Models.Vae.LtxVideoVaeDecoder"/> weight dict (no timestep conditioning).</summary>
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

    /// <summary>Builds a full <see cref="LtxVideo2Transformer"/> weight dict for the given (tiny) dual-stream config.
    /// Mirrors what the shipped checkpoints carry: the video FFN bias follows <see cref="LtxVideo2Config.FfBias"/>,
    /// while the audio and connector FFN biases and the prompt-AdaLN subtrees are present in every released LTX-2
    /// generation. <paramref name="keyframesValue"/>, when given, fills <c>keyframes_abs_pos_embedding</c> with a
    /// constant instead of noise so a test can predict its contribution.</summary>
    public static Dictionary<string, Tensor> BuildTransformer2(LtxVideo2Config c, float? keyframesValue = null)
    {
        int v = c.InnerDim, a = c.AudioInnerDim;
        int vFf = c.FfnMultiplier * v, aFf = c.FfnMultiplier * a;
        Dictionary<string, Tensor> w = new()
        {
            ["proj_in.weight"] = R([v, c.InChannels]), ["proj_in.bias"] = R([v]),
            ["audio_proj_in.weight"] = R([a, c.AudioInChannels]), ["audio_proj_in.bias"] = R([a]),
            ["proj_out.weight"] = R([c.OutChannels, v]), ["proj_out.bias"] = R([c.OutChannels]),
            ["audio_proj_out.weight"] = R([c.AudioOutChannels, a]), ["audio_proj_out.bias"] = R([c.AudioOutChannels]),
            ["scale_shift_table"] = R([c.OutputModParams, v]),
            ["audio_scale_shift_table"] = R([c.OutputModParams, a]),
        };

        AddAdaLn(w, "time_embed", v, c.SelfAttnModParams);
        AddAdaLn(w, "audio_time_embed", a, c.SelfAttnModParams);
        AddAdaLn(w, "prompt_adaln", v, 2);
        AddAdaLn(w, "audio_prompt_adaln", a, 2);
        AddAdaLn(w, "av_cross_attn_video_scale_shift", v, 4);
        AddAdaLn(w, "av_cross_attn_audio_scale_shift", a, 4);
        AddAdaLn(w, "av_cross_attn_video_a2v_gate", v, 1);
        AddAdaLn(w, "av_cross_attn_audio_v2a_gate", a, 1);

        if (c.UseKeyframesAbsPosEmbedding)
        {
            w["keyframes_abs_pos_embedding"] = keyframesValue is float kv ? Const([1, v], kv) : R([1, v]);
        }

        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"transformer_blocks.{i}";
            AddAttn(w, $"{p}.attn1", v, v, v, c.NumHeads);
            AddAttn(w, $"{p}.attn2", v, c.CrossAttentionDim, v, c.NumHeads);
            AddAttn(w, $"{p}.audio_attn1", a, a, a, c.AudioNumHeads);
            AddAttn(w, $"{p}.audio_attn2", a, c.AudioCrossAttentionDim, a, c.AudioNumHeads);
            // a2v: query is video-width, KV audio-width, output video-width (and the reverse for v2a).
            AddAttn(w, $"{p}.audio_to_video_attn", v, a, v, c.AudioNumHeads, queryOut: a);
            AddAttn(w, $"{p}.video_to_audio_attn", a, v, a, c.AudioNumHeads, queryOut: a);

            w[$"{p}.scale_shift_table"] = R([c.SelfAttnModParams, v]);
            w[$"{p}.audio_scale_shift_table"] = R([c.SelfAttnModParams, a]);
            w[$"{p}.prompt_scale_shift_table"] = R([2, v]);
            w[$"{p}.audio_prompt_scale_shift_table"] = R([2, a]);
            w[$"{p}.scale_shift_table_a2v_ca_video"] = R([5, v]);
            w[$"{p}.scale_shift_table_a2v_ca_audio"] = R([5, a]);

            w[$"{p}.ff.net.0.proj.weight"] = R([vFf, v]);
            w[$"{p}.ff.net.2.weight"] = R([v, vFf]);
            if (c.FfBias)
            {
                w[$"{p}.ff.net.0.proj.bias"] = R([vFf]);
                w[$"{p}.ff.net.2.bias"] = R([v]);
            }
            w[$"{p}.audio_ff.net.0.proj.weight"] = R([aFf, a]); w[$"{p}.audio_ff.net.0.proj.bias"] = R([aFf]);
            w[$"{p}.audio_ff.net.2.weight"] = R([a, aFf]); w[$"{p}.audio_ff.net.2.bias"] = R([a]);
        }
        return w;
    }

    private static void AddAdaLn(Dictionary<string, Tensor> w, string p, int dim, int numParams)
    {
        w[$"{p}.emb.timestep_embedder.linear_1.weight"] = R([dim, 256]);
        w[$"{p}.emb.timestep_embedder.linear_1.bias"] = R([dim]);
        w[$"{p}.emb.timestep_embedder.linear_2.weight"] = R([dim, dim]);
        w[$"{p}.emb.timestep_embedder.linear_2.bias"] = R([dim]);
        w[$"{p}.linear.weight"] = R([numParams * dim, dim]);
        w[$"{p}.linear.bias"] = R([numParams * dim]);
    }

    private static void AddAttn(Dictionary<string, Tensor> w, string p, int queryIn, int kvIn, int outDim,
        int heads, int? queryOut = null)
    {
        int qOut = queryOut ?? outDim;
        w[$"{p}.to_q.weight"] = R([qOut, queryIn]); w[$"{p}.to_q.bias"] = R([qOut]);
        w[$"{p}.to_k.weight"] = R([qOut, kvIn]); w[$"{p}.to_k.bias"] = R([qOut]);
        w[$"{p}.to_v.weight"] = R([qOut, kvIn]); w[$"{p}.to_v.bias"] = R([qOut]);
        w[$"{p}.to_out.0.weight"] = R([outDim, qOut]); w[$"{p}.to_out.0.bias"] = R([outDim]);
        w[$"{p}.q_norm.weight"] = R([qOut]);
        w[$"{p}.k_norm.weight"] = R([qOut]);
        w[$"{p}.to_gate_logits.weight"] = R([heads, queryIn]); w[$"{p}.to_gate_logits.bias"] = R([heads]);
    }

    private static Tensor Const(int[] dims, float value)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = value;
        return t;
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
