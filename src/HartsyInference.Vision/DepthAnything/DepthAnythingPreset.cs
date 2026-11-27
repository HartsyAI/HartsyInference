using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.DepthAnything;

/// <summary>Configuration for a Depth-Anything-V2 monocular relative-depth estimator (DINOv2 backbone +
/// DPT head). Mirrors the official <c>model_configs</c> table: each variant pairs a DINOv2 size with the
/// DPT head widths and the four backbone blocks the head taps.</summary>
public sealed record DepthAnythingPreset
{
    /// <summary>Hub-style identifier (e.g. <c>depth-anything/Depth-Anything-V2-Large</c>).</summary>
    public required string Name { get; init; }

    /// <summary>DINOv2 backbone configuration (no register tokens for all released variants).</summary>
    public required Dinov2Preset Backbone { get; init; }

    /// <summary>DPT fusion width (64 small, 128 base, 256 large).</summary>
    public required int Features { get; init; }

    /// <summary>Per-tap projection widths for the four reassemble branches.</summary>
    public required int[] OutChannels { get; init; }

    /// <summary>Zero-based backbone block indices whose (final-normed) outputs feed the DPT head.</summary>
    public required int[] LayerIndices { get; init; }

    /// <summary><c>depth_anything_v2_vits</c> — 24.8M params, ViT-S/14 backbone.</summary>
    public static DepthAnythingPreset Small => new()
    {
        Name = "depth-anything/Depth-Anything-V2-Small",
        Backbone = Dinov2Preset.Small,
        Features = 64, OutChannels = [48, 96, 192, 384], LayerIndices = [2, 5, 8, 11],
    };

    /// <summary><c>depth_anything_v2_vitb</c> — 97.5M params, ViT-B/14 backbone.</summary>
    public static DepthAnythingPreset Base => new()
    {
        Name = "depth-anything/Depth-Anything-V2-Base",
        Backbone = Dinov2Preset.Base,
        Features = 128, OutChannels = [96, 192, 384, 768], LayerIndices = [2, 5, 8, 11],
    };

    /// <summary><c>depth_anything_v2_vitl</c> — 335M params, ViT-L/14 backbone. The ControlNet-grade variant.</summary>
    public static DepthAnythingPreset Large => new()
    {
        Name = "depth-anything/Depth-Anything-V2-Large",
        Backbone = Dinov2Preset.Large,
        Features = 256, OutChannels = [256, 512, 1024, 1024], LayerIndices = [4, 11, 17, 23],
    };

    /// <summary>The released variant an official checkpoint was trained as, read from its backbone width
    /// (<c>pretrained.cls_token</c>) and block count. Throws for a shape no released variant has, rather than building
    /// a model the weights don't fit.</summary>
    public static DepthAnythingPreset FromWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        if (!weights.TryGetValue("pretrained.cls_token", out Tensor? cls))
        {
            throw new NotSupportedException(
                "Not a Depth-Anything-V2 checkpoint: 'pretrained.cls_token' is missing.");
        }
        int hidden = (int)cls.Shape[cls.Shape.Rank - 1];
        int blocks = 0;
        while (weights.ContainsKey($"pretrained.blocks.{blocks}.norm1.weight"))
            blocks++;

        DepthAnythingPreset? preset = hidden switch
        {
            384 => Small,
            768 => Base,
            1024 => Large,
            _ => null,
        };
        if (preset is null || preset.Backbone.NumLayers != blocks)
        {
            throw new NotSupportedException(
                $"Depth-Anything-V2 checkpoint with a {hidden}-wide, {blocks}-block backbone matches no released variant "
                + "(Small 384×12, Base 768×12, Large 1024×24).");
        }
        return preset;
    }
}
