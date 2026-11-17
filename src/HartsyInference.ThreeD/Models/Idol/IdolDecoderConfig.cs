using System.Text.Json;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>The architecture and Gaussian parameterisation of Idol's two decoders (<c>OctreeProbabilityFixedlenDecoder</c> and
/// <c>GaussianFixedlenDecoder</c>), from the checkpoint's <c>config.json</c>; the released values are the defaults.</summary>
public sealed record IdolDecoderConfig
{
    /// <summary>Width of both decoders (<c>model_channels</c>).</summary>
    public int ModelChannels { get; init; } = 1024;

    /// <summary>Latent channels the decoders cross-attend to (<c>cond_channels</c>).</summary>
    public int LatentChannels { get; init; } = 16;

    /// <summary>Attention heads (<c>num_heads</c>).</summary>
    public int Heads { get; init; } = 16;

    /// <summary>MLP width over model width (<c>mlp_ratio</c>).</summary>
    public int MlpRatio { get; init; } = 4;

    /// <summary>Blocks of the octree occupancy decoder.</summary>
    public int OctreeBlocks { get; init; } = 4;

    /// <summary>Blocks of the Gaussian decoder.</summary>
    public int GaussianBlocks { get; init; } = 16;

    /// <summary>Octree depth the anchors descend to (<c>max_voxel_level</c>).</summary>
    public int MaxVoxelLevel { get; init; } = 8;

    /// <summary>Gaussians each anchor emits (<c>num_gaussians</c>).</summary>
    public int GaussiansPerAnchor { get; init; } = 32;

    /// <summary>Learning-rate factors that multiply the raw channels (<c>lr</c>): position.</summary>
    public float XyzScale { get; init; } = 1f;

    /// <summary>Colour factor (<c>lr._features_dc</c>).</summary>
    public float ColorScale { get; init; } = 1f;

    /// <summary>Opacity factor (<c>lr._opacity</c>).</summary>
    public float OpacityScale { get; init; } = 1f;

    /// <summary>Scale factor (<c>lr._scaling</c>).</summary>
    public float ScalingScale { get; init; } = 1f;

    /// <summary>Rotation factor (<c>lr._rotation</c>).</summary>
    public float RotationScale { get; init; } = 0.1f;

    /// <summary>Size of the offset box around an anchor in anchor-cell units (<c>perturbe_size</c>).</summary>
    public float PerturbSize { get; init; } = 1.5f;

    /// <summary>The floor of the splat scale, the 3D filter kernel (<c>3d_filter_kernel_size</c>).</summary>
    public float MinKernel { get; init; } = 0.0009f;

    /// <summary>Initial splat scale (<c>scaling_bias</c>).</summary>
    public float ScalingBias { get; init; } = 0.004f;

    /// <summary>Initial splat opacity (<c>opacity_bias</c>).</summary>
    public float OpacityBias { get; init; } = 0.1f;

    public int MlpHidden => ModelChannels * MlpRatio;

    /// <summary>Channels of the Gaussian decoder's output per anchor: offset (3), colour (3), scale (3), rotation (4), opacity (1) and offset scale (1) per Gaussian.</summary>
    public int GaussianOutputChannels => GaussiansPerAnchor * 15;

    public static IdolDecoderConfig FromJson(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;
        IdolDecoderConfig c = new();
        if (root.TryGetProperty("max_voxel_level", out JsonElement level))
            c = c with { MaxVoxelLevel = level.GetInt32() };
        if (root.TryGetProperty("decoder", out JsonElement oct) && oct.TryGetProperty("args", out JsonElement oa))
        {
            c = c with
            {
                ModelChannels = Int(oa, "model_channels", c.ModelChannels), LatentChannels = Int(oa, "cond_channels", c.LatentChannels),
                Heads = Int(oa, "num_heads", c.Heads), MlpRatio = Int(oa, "mlp_ratio", c.MlpRatio), OctreeBlocks = Int(oa, "num_blocks", c.OctreeBlocks),
            };
        }

        if (root.TryGetProperty("decoder_gs", out JsonElement gs) && gs.TryGetProperty("args", out JsonElement ga))
        {
            // The two decoders share the width, heads and MLP ratio in every release; the code reads one value for both, so a config that splits them is refused.
            foreach ((string name, int expected) in new[] { ("model_channels", c.ModelChannels), ("num_heads", c.Heads), ("mlp_ratio", c.MlpRatio) })
                if (Int(ga, name, expected) != expected)
                    throw new FormatException($"The Gaussian decoder's {name} differs from the octree decoder's ({expected}); the engine builds both at one width.");
            c = c with { GaussianBlocks = Int(ga, "num_blocks", c.GaussianBlocks) };
            if (ga.TryGetProperty("representation_config", out JsonElement rep))
            {
                float F(string name, float fallback) => rep.TryGetProperty(name, out JsonElement v) ? (float)v.GetDouble() : fallback;
                c = c with
                {
                    GaussiansPerAnchor = Int(rep, "num_gaussians", c.GaussiansPerAnchor), PerturbSize = F("perturbe_size", c.PerturbSize),
                    MinKernel = F("3d_filter_kernel_size", c.MinKernel), ScalingBias = F("scaling_bias", c.ScalingBias), OpacityBias = F("opacity_bias", c.OpacityBias),
                };
                if (rep.TryGetProperty("lr", out JsonElement lr))
                {
                    float L(string name, float fallback) => lr.TryGetProperty(name, out JsonElement v) ? (float)v.GetDouble() : fallback;
                    c = c with
                    {
                        XyzScale = L("_xyz", c.XyzScale), ColorScale = L("_features_dc", c.ColorScale), OpacityScale = L("_opacity", c.OpacityScale),
                        ScalingScale = L("_scaling", c.ScalingScale), RotationScale = L("_rotation", c.RotationScale),
                    };
                }
            }
        }

        return c;
    }

    private static int Int(JsonElement e, string name, int fallback) => e.TryGetProperty(name, out JsonElement v) ? (int)v.GetDouble() : fallback;
}
