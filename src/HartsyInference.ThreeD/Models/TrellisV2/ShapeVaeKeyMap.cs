using System.Text;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>Maps upstream shape-VAE checkpoint keys to C# parameter names, and enumerates the tensors a config requires.</summary>
public static class ShapeVaeKeyMap
{
    /// <summary>Input channels of the encoder: dual-grid vertex offsets (3) and intersection flags (3).</summary>
    public const int EncoderInputChannels = 6;

    /// <summary>Output channels of the decoder: vertex sigmoids (3), intersection logits (3) and the quad-lerp scalar (1).</summary>
    public const int DecoderOutputChannels = 7;

    /// <summary>Hidden expansion of <c>SparseConvNeXtBlock3d</c>'s MLP (upstream <c>mlp_ratio</c> default).</summary>
    public const int ConvNeXtMlpRatio = 4;

    /// <summary>Converts a dotted upstream key to a C# path, e.g. <c>blocks.1.0.to_subdiv.weight</c> → <c>Blocks[1][0].ToSubdiv.Weight</c>.</summary>
    public static string ToParameterName(string checkpointKey)
    {
        StringBuilder sb = new();
        foreach (string segment in checkpointKey.Split('.'))
        {
            if (segment.Length > 0 && segment.All(char.IsAsciiDigit))
            {
                sb.Append('[').Append(segment).Append(']');
                continue;
            }
            if (sb.Length > 0) sb.Append('.');
            foreach (string word in segment.Split('_'))
                if (word.Length > 0) sb.Append(char.ToUpperInvariant(word[0])).Append(word, 1, word.Length - 1);
        }
        return sb.ToString();
    }

    /// <summary>Every tensor the encoder's checkpoint must contain for <paramref name="config"/>.</summary>
    public static IReadOnlyList<TensorSpec> Encoder(ShapeVaeEncoderConfig config)
    {
        List<TensorSpec> specs = new();
        int[] channels = config.ModelChannels;
        int levels = channels.Length;
        AddLinear(specs, "input_layer", channels[0], EncoderInputChannels);
        AddLinear(specs, "to_latent", 2 * config.LatentChannels, channels[levels - 1]);
        for (int i = 0; i < levels; i++)
        {
            for (int j = 0; j < config.NumBlocks[i]; j++) AddConvNeXt(specs, $"blocks.{i}.{j}", channels[i]);
            if (i < levels - 1) AddSpatialToChannel(specs, $"blocks.{i}.{config.NumBlocks[i]}", channels[i], channels[i + 1]);
        }
        return specs;
    }

    /// <summary>Every tensor the decoder's checkpoint must contain for <paramref name="config"/>, including the subdivision heads.</summary>
    public static IReadOnlyList<TensorSpec> Decoder(ShapeVaeDecoderConfig config)
    {
        List<TensorSpec> specs = new();
        int[] channels = config.ModelChannels;
        int levels = channels.Length;
        AddLinear(specs, "from_latent", channels[0], config.LatentChannels);
        AddLinear(specs, "output_layer", DecoderOutputChannels, channels[levels - 1]);
        for (int i = 0; i < levels; i++)
        {
            for (int j = 0; j < config.NumBlocks[i]; j++) AddConvNeXt(specs, $"blocks.{i}.{j}", channels[i]);
            if (i < levels - 1) AddChannelToSpatial(specs, $"blocks.{i}.{config.NumBlocks[i]}", channels[i], channels[i + 1]);
        }
        return specs;
    }

    private static void AddConvNeXt(List<TensorSpec> specs, string p, int c)
    {
        AddNorm(specs, $"{p}.norm", c);
        AddSparseConv(specs, $"{p}.conv", c, c);
        AddLinear(specs, $"{p}.mlp.0", c * ConvNeXtMlpRatio, c);
        AddLinear(specs, $"{p}.mlp.2", c, c * ConvNeXtMlpRatio);
    }

    private static void AddSpatialToChannel(List<TensorSpec> specs, string p, int cin, int cout)
    {
        AddNorm(specs, $"{p}.norm1", cin);
        AddSparseConv(specs, $"{p}.conv1", cin, cout / ShapeVaeConfigRules.ChildGroup);
        AddSparseConv(specs, $"{p}.conv2", cout, cout);
    }

    private static void AddChannelToSpatial(List<TensorSpec> specs, string p, int cin, int cout)
    {
        AddNorm(specs, $"{p}.norm1", cin);
        AddSparseConv(specs, $"{p}.conv1", cin, cout * ShapeVaeConfigRules.ChildGroup);
        AddSparseConv(specs, $"{p}.conv2", cout, cout);
        AddLinear(specs, $"{p}.to_subdiv", ShapeVaeConfigRules.ChildGroup, cin);
    }

    private static void AddNorm(List<TensorSpec> specs, string p, int c)
    {
        Add(specs, $"{p}.weight", c);
        Add(specs, $"{p}.bias", c);
    }

    /// <summary>A submanifold sparse conv whose module path is <paramref name="p"/>, stored in spconv's <c>[Cout, k, k, k, Cin]</c> layout.</summary>
    private static void AddSparseConv(List<TensorSpec> specs, string p, int cin, int cout)
    {
        Add(specs, $"{p}.weight", cout, 3, 3, 3, cin);
        Add(specs, $"{p}.bias", cout);
    }

    private static void AddLinear(List<TensorSpec> specs, string p, int outDim, int inDim)
    {
        Add(specs, $"{p}.weight", outDim, inDim);
        Add(specs, $"{p}.bias", outDim);
    }

    private static void Add(List<TensorSpec> specs, string key, params int[] dims)
        => specs.Add(new TensorSpec(key, ToParameterName(key), dims.Select(d => (long)d).ToArray()));

    /// <summary>One checkpoint tensor a config requires.</summary>
    /// <param name="CheckpointKey">Upstream state-dict key, relative to its encoder or decoder module.</param>
    /// <param name="ParameterName">C# parameter name, from <see cref="ToParameterName"/>.</param>
    /// <param name="Shape">Expected tensor shape.</param>
    public readonly record struct TensorSpec(string CheckpointKey, string ParameterName, long[] Shape);
}
