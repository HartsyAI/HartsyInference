using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Tests.Common;

/// <summary>Shared synthetic-weight generators for Lance + Wan2.2 VAE structural tests (image, video, decoder). One source of truth so the image and video test projects don't duplicate the weight-dict layout. Values are small random — these gate wiring/shape/finiteness, not numerics.</summary>
public static unsafe class LanceSyntheticWeights
{
    private static int _seed = 1000;

    /// <summary>Builds a full `LanceTransformer` weight dict for the given (tiny) config using the REAL checkpoint key layout: embed/vae2llm/llm2vae/latent_pos_embed/time_embedder/norms + per-layer MoT und + `_moe_gen` siblings (incl. `q_norm`/`k_norm` when the config enables QK-norm).</summary>
    public static Dictionary<string, Tensor> BuildTransformer(LanceConfig c)
    {
        int hidden = c.HiddenSize, heads = c.NumHeads, kv = c.NumKvHeads, hd = c.HeadDim, ffn = c.IntermediateSize;
        int qDim = heads * hd, kvDim = kv * hd, patch = c.PatchFeatureDim;
        Dictionary<string, Tensor> w = new()
        {
            ["embed_tokens.weight"] = R([c.VocabSize, hidden]),
            ["vae2llm.weight"] = R([hidden, patch]), ["vae2llm.bias"] = R([hidden]),
            ["llm2vae.weight"] = R([patch, hidden]), ["llm2vae.bias"] = R([patch]),
            // 4 latent frames of capacity so tiny video tests (gridT ≤ 4) index in range; the real image ckpt ships 1 frame (4096 rows).
            ["latent_pos_embed.pos_embed"] = R([4L * c.MaxLatentSize * c.MaxLatentSize, hidden]),
            ["time_embedder.mlp.0.weight"] = R([hidden, c.TimestepFrequencyDim]), ["time_embedder.mlp.0.bias"] = R([hidden]),
            ["time_embedder.mlp.2.weight"] = R([hidden, hidden]), ["time_embedder.mlp.2.bias"] = R([hidden]),
            ["norm.weight"] = R([hidden]), ["norm_moe_gen.weight"] = R([hidden]),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"layers.{i}";
            foreach (string suf in new[] { "", "_moe_gen" })
            {
                w[$"{p}.self_attn.q_proj{suf}.weight"] = R([qDim, hidden]); w[$"{p}.self_attn.q_proj{suf}.bias"] = R([qDim]);
                w[$"{p}.self_attn.k_proj{suf}.weight"] = R([kvDim, hidden]); w[$"{p}.self_attn.k_proj{suf}.bias"] = R([kvDim]);
                w[$"{p}.self_attn.v_proj{suf}.weight"] = R([kvDim, hidden]); w[$"{p}.self_attn.v_proj{suf}.bias"] = R([kvDim]);
                w[$"{p}.self_attn.o_proj{suf}.weight"] = R([hidden, qDim]);
                if (c.QkNorm)
                {
                    w[$"{p}.self_attn.q_norm{suf}.weight"] = R([hd]);
                    w[$"{p}.self_attn.k_norm{suf}.weight"] = R([hd]);
                }
                w[$"{p}.mlp{suf}.gate_proj.weight"] = R([ffn, hidden]);
                w[$"{p}.mlp{suf}.up_proj.weight"] = R([ffn, hidden]);
                w[$"{p}.mlp{suf}.down_proj.weight"] = R([hidden, ffn]);
            }
            w[$"{p}.input_layernorm.weight"] = R([hidden]);
            w[$"{p}.post_attention_layernorm.weight"] = R([hidden]);
            w[$"{p}.input_layernorm_moe_gen.weight"] = R([hidden]);
            w[$"{p}.post_attention_layernorm_moe_gen.weight"] = R([hidden]);
        }
        return w;
    }

    /// <summary>Builds a full `Wan22VaeDecoder` weight dict (conv2 + decoder.conv1 + middle Res/Attn/Res + up-stages with resample/time_conv + head).</summary>
    public static Dictionary<string, Tensor> BuildVae(int dim, int zDim, int[] dimMult, int numResBlocks, bool[] tUp)
    {
        Dictionary<string, Tensor> w = new();
        int[] dims = new int[dimMult.Length + 1];
        dims[0] = dim * dimMult[^1];
        for (int i = 0; i < dimMult.Length; i++) dims[i + 1] = dim * dimMult[dimMult.Length - 1 - i];

        w["conv2.weight"] = R([zDim, zDim, 1, 1, 1]); w["conv2.bias"] = R([zDim]);
        w["decoder.conv1.weight"] = R([dims[0], zDim, 3, 3, 3]); w["decoder.conv1.bias"] = R([dims[0]]);
        AddRes(w, "decoder.middle.0", dims[0], dims[0]);
        AddAttn(w, "decoder.middle.1", dims[0]);
        AddRes(w, "decoder.middle.2", dims[0], dims[0]);
        int mult = numResBlocks + 1;
        for (int i = 0; i < dimMult.Length; i++)
        {
            int cur = dims[i], outDim = dims[i + 1];
            for (int j = 0; j < mult; j++) { AddRes(w, $"decoder.upsamples.{i}.upsamples.{j}", cur, outDim); cur = outDim; }
            if (i != dimMult.Length - 1)
            {
                w[$"decoder.upsamples.{i}.upsamples.{mult}.resample.1.weight"] = R([outDim, outDim, 3, 3]);
                w[$"decoder.upsamples.{i}.upsamples.{mult}.resample.1.bias"] = R([outDim]);
                if (i < tUp.Length && tUp[i])
                {
                    w[$"decoder.upsamples.{i}.upsamples.{mult}.time_conv.weight"] = R([2 * outDim, outDim, 3, 1, 1]);
                    w[$"decoder.upsamples.{i}.upsamples.{mult}.time_conv.bias"] = R([2 * outDim]);
                }
            }
        }
        w["decoder.head.0.gamma"] = R([dims[^1]]);
        w["decoder.head.2.weight"] = R([12, dims[^1], 3, 3, 3]); w["decoder.head.2.bias"] = R([12]);
        return w;
    }

    /// <summary>Builds a full `Wan22VaeEncoder` weight dict (encoder.conv1 + down-stages with resample/time_conv + middle Res/Attn/Res + head + top-level quant conv1).</summary>
    public static Dictionary<string, Tensor> BuildVaeEncoder(int dim, int zDim, int[] dimMult, int numResBlocks, bool[] tDown)
    {
        Dictionary<string, Tensor> w = new();
        int[] dims = new int[dimMult.Length + 1];
        dims[0] = dim;
        for (int i = 0; i < dimMult.Length; i++) dims[i + 1] = dim * dimMult[i];

        w["encoder.conv1.weight"] = R([dims[0], 12, 3, 3, 3]); w["encoder.conv1.bias"] = R([dims[0]]);
        for (int i = 0; i < dimMult.Length; i++)
        {
            int cur = dims[i], outDim = dims[i + 1];
            for (int j = 0; j < numResBlocks; j++) { AddRes(w, $"encoder.downsamples.{i}.downsamples.{j}", cur, outDim); cur = outDim; }
            if (i != dimMult.Length - 1)
            {
                w[$"encoder.downsamples.{i}.downsamples.{numResBlocks}.resample.1.weight"] = R([outDim, outDim, 3, 3]);
                w[$"encoder.downsamples.{i}.downsamples.{numResBlocks}.resample.1.bias"] = R([outDim]);
                if (i < tDown.Length && tDown[i])
                {
                    w[$"encoder.downsamples.{i}.downsamples.{numResBlocks}.time_conv.weight"] = R([outDim, outDim, 3, 1, 1]);
                    w[$"encoder.downsamples.{i}.downsamples.{numResBlocks}.time_conv.bias"] = R([outDim]);
                }
            }
        }
        AddRes(w, "encoder.middle.0", dims[^1], dims[^1]);
        AddAttn(w, "encoder.middle.1", dims[^1]);
        AddRes(w, "encoder.middle.2", dims[^1], dims[^1]);
        w["encoder.head.0.gamma"] = R([dims[^1]]);
        w["encoder.head.2.weight"] = R([2 * zDim, dims[^1], 3, 3, 3]); w["encoder.head.2.bias"] = R([2 * zDim]);
        w["conv1.weight"] = R([2 * zDim, 2 * zDim, 1, 1, 1]); w["conv1.bias"] = R([2 * zDim]);
        return w;
    }

    private static void AddRes(Dictionary<string, Tensor> w, string p, int inDim, int outDim)
    {
        w[$"{p}.residual.0.gamma"] = R([inDim]);
        w[$"{p}.residual.2.weight"] = R([outDim, inDim, 3, 3, 3]); w[$"{p}.residual.2.bias"] = R([outDim]);
        w[$"{p}.residual.3.gamma"] = R([outDim]);
        w[$"{p}.residual.6.weight"] = R([outDim, outDim, 3, 3, 3]); w[$"{p}.residual.6.bias"] = R([outDim]);
        if (inDim != outDim) { w[$"{p}.shortcut.weight"] = R([outDim, inDim, 1, 1, 1]); w[$"{p}.shortcut.bias"] = R([outDim]); }
    }

    private static void AddAttn(Dictionary<string, Tensor> w, string p, int dim)
    {
        w[$"{p}.norm.gamma"] = R([dim]);
        w[$"{p}.to_qkv.weight"] = R([3 * dim, dim, 1, 1]); w[$"{p}.to_qkv.bias"] = R([3 * dim]);
        w[$"{p}.proj.weight"] = R([dim, dim, 1, 1]); w[$"{p}.proj.bias"] = R([dim]);
    }

    private static Tensor R(long[] dims)
    {
        Tensor t = new Tensor(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }
}
