using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Shape-derived contract for the published five-block MiniMax-H3 Fun ControlNet-Union branch.</summary>
public sealed record MiniMaxH3FunControlConfig
{
    /// <summary>Residual-stream width shared with the base H3 transformer.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of control blocks; the published branch carries five.</summary>
    public required int NumBlocks { get; init; }

    /// <summary>Attention head count derived from the fused QKV rows.</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>Per-head attention width.</summary>
    public required int AttentionHeadDim { get; init; }

    /// <summary>SwiGLU hidden width after folding its two input halves.</summary>
    public required int FfnHiddenSize { get; init; }

    /// <summary>Timestep-coordinate width consumed by every control AdaLN projection.</summary>
    public required int TimeEmbedDim { get; init; }

    /// <summary>Channels before the 1x2x2 patchifier; 49 means control24 + visibility1 + source24.</summary>
    public required int ControlInputChannels { get; init; }

    /// <summary>Patched control-row width accepted by <c>control_proj_in</c>.</summary>
    public int ControlPatchDim => ControlInputChannels * 4;

    /// <summary>Main transformer layers addressed by control blocks in order.</summary>
    public IReadOnlyList<int> InjectionLayers { get; init; } = [0, 10, 20, 30, 40];

    /// <summary>Detects and fully validates a branch header before any forward can begin.</summary>
    public static MiniMaxH3FunControlConfig Detect(IReadOnlyDictionary<string, Tensor> weights,
        bool requirePublishedLayout = true)
    {
        ArgumentNullException.ThrowIfNull(weights);
        Tensor projection = Require(weights, "control_proj_in.weight");
        if (projection.Shape.Rank != 2 || projection.Shape[0] <= 0 || projection.Shape[1] <= 0
            || projection.Shape[1] % 4 != 0)
        {
            throw new HartsyInferenceException(
                $"MiniMax-H3 control_proj_in.weight must be [hidden,channels*4], got {projection.Shape}.");
        }
        int hidden = checked((int)projection.Shape[0]);
        int channels = checked((int)projection.Shape[1] / 4);
        int blocks = CountBlocks(weights);
        if (blocks == 0)
        {
            throw new HartsyInferenceException("MiniMax-H3 Fun ControlNet has no contiguous control_blocks.* stack.");
        }

        Tensor qkv = Require(weights, "control_blocks.0.attn.qkv_proj.weight");
        Tensor qNorm = Require(weights, "control_blocks.0.attn.q_norm.weight");
        Tensor fc1 = Require(weights, "control_blocks.0.mlp.fc1.weight");
        Tensor adaln = Require(weights, "control_blocks.0.adaln_proj.linear.weight");
        if (qkv.Shape.Rank != 2 || qNorm.Shape.Rank != 1 || qNorm.Shape[0] <= 0
            || qkv.Shape[0] % (3 * qNorm.Shape[0]) != 0 || fc1.Shape.Rank != 2 || fc1.Shape[0] % 2 != 0
            || adaln.Shape.Rank != 2)
        {
            throw new HartsyInferenceException(
                $"Invalid MiniMax-H3 control block-0 geometry: qkv={qkv.Shape}, qNorm={qNorm.Shape}, "
                + $"fc1={fc1.Shape}, adaln={adaln.Shape}.");
        }
        int headDim = checked((int)qNorm.Shape[0]);
        int heads = checked((int)(qkv.Shape[0] / (3 * headDim)));
        int ffn = checked((int)(fc1.Shape[0] / 2));
        int time = checked((int)adaln.Shape[1]);

        MiniMaxH3FunControlConfig config = new MiniMaxH3FunControlConfig
        {
            HiddenSize = hidden,
            NumBlocks = blocks,
            NumAttentionHeads = heads,
            AttentionHeadDim = headDim,
            FfnHiddenSize = ffn,
            TimeEmbedDim = time,
            ControlInputChannels = channels,
            InjectionLayers = Enumerable.Range(0, blocks).Select(index => index * 10).ToArray(),
        };
        if (requirePublishedLayout && (blocks != 5 || channels != 49))
        {
            throw new HartsyInferenceException(
                $"The published MiniMax-H3 Fun branch requires five blocks and 49 control channels; "
                + $"got blocks={blocks}, channels={channels}.");
        }
        ValidateAll(weights, config);
        return config;
    }

    /// <summary>Refuses a branch whose stream, attention, MLP, or AdaLN geometry differs from its base transformer.</summary>
    public void ValidateBase(MiniMaxH3Config baseConfig)
    {
        ArgumentNullException.ThrowIfNull(baseConfig);
        if (HiddenSize != baseConfig.HiddenSize || NumAttentionHeads != baseConfig.NumAttentionHeads
            || AttentionHeadDim != baseConfig.AttentionHeadDim || FfnHiddenSize != baseConfig.FfnHiddenSize
            || TimeEmbedDim != baseConfig.TimeEmbedDim)
        {
            throw new HartsyInferenceException(
                $"MiniMax-H3 Fun branch does not match its base: control hidden/heads/headDim/ffn/time="
                + $"{HiddenSize}/{NumAttentionHeads}/{AttentionHeadDim}/{FfnHiddenSize}/{TimeEmbedDim}, base="
                + $"{baseConfig.HiddenSize}/{baseConfig.NumAttentionHeads}/{baseConfig.AttentionHeadDim}/"
                + $"{baseConfig.FfnHiddenSize}/{baseConfig.TimeEmbedDim}.");
        }
        if (InjectionLayers.Count != NumBlocks || InjectionLayers.Count == 0
            || InjectionLayers[^1] >= baseConfig.NumLayers)
        {
            throw new HartsyInferenceException(
                $"MiniMax-H3 Fun injection layers [{string.Join(',', InjectionLayers)}] do not fit a "
                + $"{baseConfig.NumLayers}-block base.");
        }
    }

    private static void ValidateAll(IReadOnlyDictionary<string, Tensor> weights, MiniMaxH3FunControlConfig config)
    {
        int hidden = config.HiddenSize;
        int inner = config.NumAttentionHeads * config.AttentionHeadDim;
        RequireShape(weights, "control_proj_in.weight", hidden, config.ControlPatchDim);
        RequireShape(weights, "control_proj_in.bias", hidden);
        for (int index = 0; index < config.NumBlocks; index++)
        {
            string prefix = $"control_blocks.{index}";
            RequireShape(weights, prefix + ".norm1.weight", hidden);
            RequireShape(weights, prefix + ".norm2.weight", hidden);
            RequireShape(weights, prefix + ".attn.qkv_proj.weight", inner * 3, hidden);
            RequireShape(weights, prefix + ".attn.q_norm.weight", config.AttentionHeadDim);
            RequireShape(weights, prefix + ".attn.k_norm.weight", config.AttentionHeadDim);
            RequireShape(weights, prefix + ".attn.out_proj.weight", hidden, inner);
            RequireShape(weights, prefix + ".mlp.fc1.weight", config.FfnHiddenSize * 2, hidden);
            RequireShape(weights, prefix + ".mlp.fc2.weight", hidden, config.FfnHiddenSize);
            RequireShape(weights, prefix + ".adaln_proj.linear.weight", hidden * 18, config.TimeEmbedDim);
            RequireShape(weights, prefix + ".adaln_proj.linear.bias", hidden * 18);
            RequireShape(weights, prefix + ".after_proj.weight", hidden, hidden);
            RequireShape(weights, prefix + ".after_proj.bias", hidden);
            if (index == 0)
            {
                RequireShape(weights, prefix + ".before_proj.weight", hidden, hidden);
                RequireShape(weights, prefix + ".before_proj.bias", hidden);
            }
        }
    }

    private static int CountBlocks(IReadOnlyDictionary<string, Tensor> weights)
    {
        HashSet<int> indices = new HashSet<int>();
        foreach (string key in weights.Keys)
        {
            if (!key.StartsWith("control_blocks.", StringComparison.Ordinal))
            {
                continue;
            }
            int dot = key.IndexOf('.', "control_blocks.".Length);
            if (dot > "control_blocks.".Length
                && int.TryParse(key.AsSpan("control_blocks.".Length, dot - "control_blocks.".Length), out int index))
            {
                indices.Add(index);
            }
        }
        int count = 0;
        while (indices.Contains(count))
        {
            count++;
        }
        if (indices.Count != count)
        {
            throw new HartsyInferenceException(
                $"MiniMax-H3 control block indices must be contiguous from zero; found [{string.Join(',', indices.Order())}].");
        }
        return count;
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        return weights.TryGetValue(key, out Tensor? tensor) ? tensor
            : throw new HartsyInferenceException($"MiniMax-H3 Fun ControlNet is missing '{key}'.");
    }

    private static void RequireShape(IReadOnlyDictionary<string, Tensor> weights, string key, params long[] shape)
    {
        Tensor tensor = Require(weights, key);
        TensorShape expected = new TensorShape(shape);
        if (tensor.Shape != expected)
        {
            throw new HartsyInferenceException(
                $"MiniMax-H3 Fun ControlNet tensor '{key}' has shape {tensor.Shape}, expected {expected}.");
        }
    }
}
