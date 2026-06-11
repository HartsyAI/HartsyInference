using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;

namespace SharpInference.Tests.Common;

/// <summary>Synthetic weights for <see cref="MatrixGame2Transformer"/> (Wan core + I2V img_emb projection + per-block
/// ActionModules with the variant keyboard dims) and the Wan2.1 VAE decoder/encoder.</summary>
public static unsafe class MatrixGame2SyntheticWeights
{
    private static int s_seed = 15000;

    public static Dictionary<string, Tensor> BuildTransformer(MatrixGame2Config c)
    {
        int dim = c.InnerDim;
        int winLen = c.VaeTemporalCompression * c.ActionWindowSize;
        Dictionary<string, Tensor> w = WanSyntheticWeights.BuildTransformer(c.ToWanConfig());

        // I2V image projection (img_emb MLPProj → diffusers condition_embedder.image_embedder.*).
        w["condition_embedder.image_embedder.norm1.weight"] = R([c.ClipContextDim]); w["condition_embedder.image_embedder.norm1.bias"] = R([c.ClipContextDim]);
        w["condition_embedder.image_embedder.ff.net.0.proj.weight"] = R([dim, c.ClipContextDim]); w["condition_embedder.image_embedder.ff.net.0.proj.bias"] = R([dim]);
        w["condition_embedder.image_embedder.ff.net.2.weight"] = R([dim, dim]); w["condition_embedder.image_embedder.ff.net.2.bias"] = R([dim]);
        w["condition_embedder.image_embedder.norm2.weight"] = R([dim]); w["condition_embedder.image_embedder.norm2.bias"] = R([dim]);

        int headDim = c.ActionStreamDim / c.ActionHeads;
        IEnumerable<int> blocks = c.ActionBlocks ?? Enumerable.Range(0, c.NumLayers);
        foreach (int i in blocks)
        {
            string p = $"blocks.{i}.action_model";
            if (c.EnableMouse)
            {
                w[$"{p}.mouse_mlp.0.weight"] = R([c.ActionStreamDim, dim + 2 * winLen]); w[$"{p}.mouse_mlp.0.bias"] = R([c.ActionStreamDim]);
                w[$"{p}.mouse_mlp.2.weight"] = R([c.ActionStreamDim, c.ActionStreamDim]); w[$"{p}.mouse_mlp.2.bias"] = R([c.ActionStreamDim]);
                w[$"{p}.mouse_mlp.3.weight"] = R([c.ActionStreamDim]); w[$"{p}.mouse_mlp.3.bias"] = R([c.ActionStreamDim]);
                w[$"{p}.t_qkv.weight"] = R([3 * c.ActionStreamDim, c.ActionStreamDim]);
                w[$"{p}.proj_mouse.weight"] = R([dim, c.ActionStreamDim]); w[$"{p}.proj_mouse.bias"] = R([dim]);
                w[$"{p}.img_attn_q_norm.weight"] = R([headDim]); w[$"{p}.img_attn_k_norm.weight"] = R([headDim]);
            }
            w[$"{p}.keyboard_embed.0.weight"] = R([c.ActionHiddenSize, c.KeyboardDim]); w[$"{p}.keyboard_embed.0.bias"] = R([c.ActionHiddenSize]);
            w[$"{p}.keyboard_embed.2.weight"] = R([c.ActionHiddenSize, c.ActionHiddenSize]); w[$"{p}.keyboard_embed.2.bias"] = R([c.ActionHiddenSize]);
            w[$"{p}.keyboard_attn_kv.weight"] = R([2 * c.ActionStreamDim, c.ActionHiddenSize * winLen]);
            w[$"{p}.mouse_attn_q.weight"] = R([c.ActionStreamDim, dim]);
            w[$"{p}.proj_keyboard.weight"] = R([dim, c.ActionStreamDim]); w[$"{p}.proj_keyboard.bias"] = R([dim]);
            w[$"{p}.key_attn_q_norm.weight"] = R([headDim]); w[$"{p}.key_attn_k_norm.weight"] = R([headDim]);
        }
        return w;
    }

    /// <summary>Wan2.1 decoder dict (flat <c>decoder.upsamples.{idx}</c>, channel-halving resamples).</summary>
    public static Dictionary<string, Tensor> BuildVae21Decoder(int dim, int zDim, int[] dimMult, int numResBlocks, bool[] tUp)
    {
        Dictionary<string, Tensor> w = new();
        int[] dims = new int[dimMult.Length + 1];
        dims[0] = dim * dimMult[^1];
        for (int i = 0; i < dimMult.Length; i++) dims[i + 1] = dim * dimMult[dimMult.Length - 1 - i];

        w["conv2.weight"] = R([zDim, zDim, 1, 1, 1]); w["conv2.bias"] = R([zDim]);
        w["decoder.conv1.weight"] = R([dims[0], zDim, 3, 3, 3]); w["decoder.conv1.bias"] = R([dims[0]]);
        AddRes21(w, "decoder.middle.0", dims[0], dims[0]);
        AddAttn21(w, "decoder.middle.1", dims[0]);
        AddRes21(w, "decoder.middle.2", dims[0], dims[0]);

        int flat = 0;
        for (int i = 0; i < dimMult.Length; i++)
        {
            int inDim = dims[i], outDim = dims[i + 1];
            if (i is 1 or 2 or 3) inDim /= 2;
            int cur = inDim;
            for (int j = 0; j < numResBlocks + 1; j++) { AddRes21(w, $"decoder.upsamples.{flat++}", cur, outDim); cur = outDim; }
            if (i != dimMult.Length - 1)
            {
                w[$"decoder.upsamples.{flat}.resample.1.weight"] = R([outDim / 2, outDim, 3, 3]);
                w[$"decoder.upsamples.{flat}.resample.1.bias"] = R([outDim / 2]);
                if (i < tUp.Length && tUp[i])
                {
                    w[$"decoder.upsamples.{flat}.time_conv.weight"] = R([2 * outDim, outDim, 3, 1, 1]);
                    w[$"decoder.upsamples.{flat}.time_conv.bias"] = R([2 * outDim]);
                }
                flat++;
            }
        }
        w["decoder.head.0.gamma"] = R([dims[^1]]);
        w["decoder.head.2.weight"] = R([3, dims[^1], 3, 3, 3]); w["decoder.head.2.bias"] = R([3]);
        return w;
    }

    /// <summary>Wan2.1 encoder dict (flat <c>encoder.downsamples.{idx}</c>, same-channel stride-2 downsamples).</summary>
    public static Dictionary<string, Tensor> BuildVae21Encoder(int dim, int zDim, int[] dimMult, int numResBlocks, bool[] tDown)
    {
        Dictionary<string, Tensor> w = new();
        int[] dims = new int[dimMult.Length + 1];
        dims[0] = dim;
        for (int i = 0; i < dimMult.Length; i++) dims[i + 1] = dim * dimMult[i];

        w["encoder.conv1.weight"] = R([dims[0], 3, 3, 3, 3]); w["encoder.conv1.bias"] = R([dims[0]]);
        int flat = 0;
        for (int i = 0; i < dimMult.Length; i++)
        {
            int cur = dims[i], outDim = dims[i + 1];
            for (int j = 0; j < numResBlocks; j++) { AddRes21(w, $"encoder.downsamples.{flat++}", cur, outDim); cur = outDim; }
            if (i != dimMult.Length - 1)
            {
                w[$"encoder.downsamples.{flat}.resample.1.weight"] = R([outDim, outDim, 3, 3]);
                w[$"encoder.downsamples.{flat}.resample.1.bias"] = R([outDim]);
                if (i < tDown.Length && tDown[i])
                {
                    w[$"encoder.downsamples.{flat}.time_conv.weight"] = R([outDim, outDim, 3, 1, 1]);
                    w[$"encoder.downsamples.{flat}.time_conv.bias"] = R([outDim]);
                }
                flat++;
            }
        }
        AddRes21(w, "encoder.middle.0", dims[^1], dims[^1]);
        AddAttn21(w, "encoder.middle.1", dims[^1]);
        AddRes21(w, "encoder.middle.2", dims[^1], dims[^1]);
        w["encoder.head.0.gamma"] = R([dims[^1]]);
        w["encoder.head.2.weight"] = R([2 * zDim, dims[^1], 3, 3, 3]); w["encoder.head.2.bias"] = R([2 * zDim]);
        w["conv1.weight"] = R([2 * zDim, 2 * zDim, 1, 1, 1]); w["conv1.bias"] = R([2 * zDim]);
        return w;
    }

    private static void AddRes21(Dictionary<string, Tensor> w, string p, int inDim, int outDim)
    {
        w[$"{p}.residual.0.gamma"] = R([inDim]);
        w[$"{p}.residual.2.weight"] = R([outDim, inDim, 3, 3, 3]); w[$"{p}.residual.2.bias"] = R([outDim]);
        w[$"{p}.residual.3.gamma"] = R([outDim]);
        w[$"{p}.residual.6.weight"] = R([outDim, outDim, 3, 3, 3]); w[$"{p}.residual.6.bias"] = R([outDim]);
        if (inDim != outDim) { w[$"{p}.shortcut.weight"] = R([outDim, inDim, 1, 1, 1]); w[$"{p}.shortcut.bias"] = R([outDim]); }
    }

    private static void AddAttn21(Dictionary<string, Tensor> w, string p, int dim)
    {
        w[$"{p}.norm.gamma"] = R([dim]);
        w[$"{p}.to_qkv.weight"] = R([3 * dim, dim, 1, 1]); w[$"{p}.to_qkv.bias"] = R([3 * dim]);
        w[$"{p}.proj.weight"] = R([dim, dim, 1, 1]); w[$"{p}.proj.bias"] = R([dim]);
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
