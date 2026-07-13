using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Tests.Common;

/// <summary>Synthetic weights for <see cref="MatrixGame3Transformer"/>: the Wan core (via
/// <see cref="WanSyntheticWeights.BuildTransformer"/>) plus per-block ActionModule weights and the Plücker projection.</summary>
public static unsafe class MatrixGame3SyntheticWeights
{
    private static int _seed = 9000;

    public static Dictionary<string, Tensor> Build(MatrixGame3Config c,
        int actionStreamDim = 0, int actionHiddenSize = 0, int actionHeads = 0, int pluckerPatchDim = 0)
    {
        if (actionStreamDim <= 0) actionStreamDim = c.ActionStreamDim;
        if (actionHiddenSize <= 0) actionHiddenSize = c.ActionHiddenSize;
        if (actionHeads <= 0) actionHeads = c.ActionHeads;
        if (pluckerPatchDim <= 0) pluckerPatchDim = c.PluckerPatchDim;
        int dim = c.InnerDim;
        int winLen = c.VaeTemporalCompression * c.ActionWindowSize;

        Dictionary<string, Tensor> w = WanSyntheticWeights.BuildTransformer(c.ToWanConfig());

        w["patch_embedding_wancamctrl.weight"] = R([dim, pluckerPatchDim]);
        w["patch_embedding_wancamctrl.bias"] = R([dim]);
        w["c2ws_hidden_states_layer1.weight"] = R([dim, dim]); w["c2ws_hidden_states_layer1.bias"] = R([dim]);
        w["c2ws_hidden_states_layer2.weight"] = R([dim, dim]); w["c2ws_hidden_states_layer2.bias"] = R([dim]);

        int headDim = actionStreamDim / actionHeads;
        IEnumerable<int> blocks = c.ActionBlocks ?? Enumerable.Range(0, c.NumLayers);
        foreach (int i in blocks)
        {
            string p = $"blocks.{i}.action_model";
            w[$"{p}.mouse_mlp.0.weight"] = R([actionStreamDim, dim + 2 * winLen]); w[$"{p}.mouse_mlp.0.bias"] = R([actionStreamDim]);
            w[$"{p}.mouse_mlp.2.weight"] = R([actionStreamDim, actionStreamDim]); w[$"{p}.mouse_mlp.2.bias"] = R([actionStreamDim]);
            w[$"{p}.mouse_mlp.3.weight"] = R([actionStreamDim]); w[$"{p}.mouse_mlp.3.bias"] = R([actionStreamDim]);
            w[$"{p}.t_qkv.weight"] = R([3 * actionStreamDim, actionStreamDim]); w[$"{p}.t_qkv.bias"] = R([3 * actionStreamDim]);
            w[$"{p}.proj_mouse.weight"] = R([dim, actionStreamDim]); w[$"{p}.proj_mouse.bias"] = R([dim]);
            w[$"{p}.img_attn_q_norm.weight"] = R([headDim]); w[$"{p}.img_attn_k_norm.weight"] = R([headDim]);
            w[$"{p}.keyboard_embed.0.weight"] = R([actionHiddenSize, 6]); w[$"{p}.keyboard_embed.0.bias"] = R([actionHiddenSize]);
            w[$"{p}.keyboard_embed.2.weight"] = R([actionHiddenSize, actionHiddenSize]); w[$"{p}.keyboard_embed.2.bias"] = R([actionHiddenSize]);
            w[$"{p}.keyboard_attn_kv.weight"] = R([2 * actionStreamDim, actionHiddenSize * winLen]); w[$"{p}.keyboard_attn_kv.bias"] = R([2 * actionStreamDim]);
            w[$"{p}.mouse_attn_q.weight"] = R([actionStreamDim, dim]); w[$"{p}.mouse_attn_q.bias"] = R([actionStreamDim]);
            w[$"{p}.proj_keyboard.weight"] = R([dim, actionStreamDim]); w[$"{p}.proj_keyboard.bias"] = R([dim]);
            w[$"{p}.key_attn_q_norm.weight"] = R([headDim]); w[$"{p}.key_attn_k_norm.weight"] = R([headDim]);
        }
        // Per-block Plücker camera injection (cam_* layers) — present on every block in the real use_memory checkpoint.
        for (int i = 0; i < c.NumLayers; i++)
        {
            string b = $"blocks.{i}";
            w[$"{b}.cam_injector_layer1.weight"] = R([dim, dim]); w[$"{b}.cam_injector_layer1.bias"] = R([dim]);
            w[$"{b}.cam_injector_layer2.weight"] = R([dim, dim]); w[$"{b}.cam_injector_layer2.bias"] = R([dim]);
            w[$"{b}.cam_scale_layer.weight"] = R([dim, dim]); w[$"{b}.cam_scale_layer.bias"] = R([dim]);
            w[$"{b}.cam_shift_layer.weight"] = R([dim, dim]); w[$"{b}.cam_shift_layer.bias"] = R([dim]);
        }
        return w;
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
