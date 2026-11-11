namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>Enumerates the tensors a TRELLIS.2 flow checkpoint holds for a <see cref="Trellis2FlowConfig"/>, under the upstream
/// state-dict names, which the flow classes read directly.</summary>
public static class Trellis2FlowKeyMap
{
    /// <summary>Every tensor the checkpoint must contain for <paramref name="config"/>, with its expected shape.</summary>
    public static IReadOnlyList<ShapeVaeKeyMap.TensorSpec> Specs(Trellis2FlowConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        int c = config.ModelChannels;
        List<ShapeVaeKeyMap.TensorSpec> specs = [];
        AddLinear(specs, "input_layer", c, config.InChannels);
        AddLinear(specs, "t_embedder.mlp.0", c, Trellis2FlowConfig.TimestepFrequencyDim);
        AddLinear(specs, "t_embedder.mlp.2", c, c);
        AddLinear(specs, "adaLN_modulation.1", 6 * c, c);
        AddLinear(specs, "out_layer", config.OutChannels, c);
        for (int i = 0; i < config.NumBlocks; i++)
            AddBlock(specs, config, $"blocks.{i}");
        return specs;
    }

    private static void AddBlock(List<ShapeVaeKeyMap.TensorSpec> specs, Trellis2FlowConfig config, string p)
    {
        int c = config.ModelChannels;
        Add(specs, $"{p}.modulation", 6 * c);
        AddLinear(specs, $"{p}.self_attn.to_qkv", 3 * c, c);
        if (config.QkRmsNorm)
            AddGammas(specs, $"{p}.self_attn", config);
        AddLinear(specs, $"{p}.self_attn.to_out", c, c);
        Add(specs, $"{p}.norm2.weight", c);
        Add(specs, $"{p}.norm2.bias", c);
        AddLinear(specs, $"{p}.cross_attn.to_q", c, c);
        AddLinear(specs, $"{p}.cross_attn.to_kv", 2 * c, config.CondChannels);
        if (config.QkRmsNormCross)
            AddGammas(specs, $"{p}.cross_attn", config);
        AddLinear(specs, $"{p}.cross_attn.to_out", c, c);
        AddLinear(specs, $"{p}.mlp.mlp.0", config.MlpHidden, c);
        AddLinear(specs, $"{p}.mlp.mlp.2", c, config.MlpHidden);
    }

    private static void AddGammas(List<ShapeVaeKeyMap.TensorSpec> specs, string p, Trellis2FlowConfig config)
    {
        Add(specs, $"{p}.q_rms_norm.gamma", config.NumHeads, config.HeadDim);
        Add(specs, $"{p}.k_rms_norm.gamma", config.NumHeads, config.HeadDim);
    }

    private static void AddLinear(List<ShapeVaeKeyMap.TensorSpec> specs, string p, int outDim, int inDim)
    {
        Add(specs, $"{p}.weight", outDim, inDim);
        Add(specs, $"{p}.bias", outDim);
    }

    private static void Add(List<ShapeVaeKeyMap.TensorSpec> specs, string key, params int[] dims)
        => specs.Add(new ShapeVaeKeyMap.TensorSpec(key, ShapeVaeKeyMap.ToParameterName(key), dims.Select(d => (long)d).ToArray()));
}
