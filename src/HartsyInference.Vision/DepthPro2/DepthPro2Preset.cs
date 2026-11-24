using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.DepthPro2;

/// <summary>
/// One Depth Pro 2 size, as each <c>apple/DepthPro2-*</c> repo's <c>config.json</c> states it: the timm DINOv3 backbone, the blocks the pyramid taps, the
/// blocks that attend across frames, the four reassembled widths and the decoder width. The backbone has no q/k/v bias, no final norm, float32 RoPE periods and
/// RoPE coordinates normalised by the longer grid side (timm's <c>normalize_coords = "max"</c>).
/// </summary>
public sealed record DepthPro2Preset
{
    /// <summary>The input height and width must be multiples of this; the image is replicate-padded up to it.</summary>
    public const int InputMultiple = 32;

    /// <summary>The catalog-style id, such as <c>depthpro2-s</c>.</summary>
    public required string Name { get; init; }

    /// <summary>The Hugging Face repo holding the F32 safetensors.</summary>
    public required string Repo { get; init; }

    /// <summary>The DINOv3 backbone, read from the checkpoint's <c>encoder.*</c> timm keys.</summary>
    public required Dinov2Preset Backbone { get; init; }

    /// <summary>The blocks whose patch tokens feed the pyramid, at 1/4, 1/8, 1/16 and 1/32 resolution in that order.</summary>
    public required int[] ReassembledBlocks { get; init; }

    /// <summary>The blocks that run the frames of a clip as one sequence; a lone frame keeps the prefix tokens it went in with.</summary>
    public required int[] CrossFrameBlocks { get; init; }

    /// <summary>The channel count of each pyramid level, highest resolution first.</summary>
    public required int[] DimsReassemble { get; init; }

    /// <summary>The channel count of every decoder stage.</summary>
    public required int DimDecoder { get; init; }

    /// <summary><c>apple/DepthPro2-S</c>: ViT-S/16.</summary>
    public static DepthPro2Preset Small => new()
    {
        Name = "depthpro2-s", Repo = "apple/DepthPro2-S", Backbone = TimmBackbone(Dinov2Preset.V3Small16),
        ReassembledBlocks = [2, 5, 8, 11], CrossFrameBlocks = [5, 7, 9, 11], DimsReassemble = [48, 96, 192, 384], DimDecoder = 64,
    };

    /// <summary><c>apple/DepthPro2-B</c>: ViT-B/16.</summary>
    public static DepthPro2Preset Base => new()
    {
        Name = "depthpro2-b", Repo = "apple/DepthPro2-B", Backbone = TimmBackbone(Dinov2Preset.V3Base16),
        ReassembledBlocks = [2, 5, 8, 11], CrossFrameBlocks = [5, 7, 9, 11], DimsReassemble = [96, 192, 384, 768], DimDecoder = 128,
    };

    /// <summary><c>apple/DepthPro2-L</c>: ViT-L/16.</summary>
    public static DepthPro2Preset Large => new()
    {
        Name = "depthpro2-l", Repo = "apple/DepthPro2-L", Backbone = TimmBackbone(Dinov2Preset.V3Large16),
        ReassembledBlocks = [5, 11, 17, 23], CrossFrameBlocks = [9, 11, 13, 15, 17, 19, 21, 23], DimsReassemble = [256, 512, 1024, 1024], DimDecoder = 256,
    };

    /// <summary><c>apple/DepthPro2-H</c>: ViT-H+/16 with the SwiGLU MLP.</summary>
    public static DepthPro2Preset HugePlus => new()
    {
        Name = "depthpro2-h", Repo = "apple/DepthPro2-H", Backbone = TimmBackbone(Dinov2Preset.V3HugePlus16),
        ReassembledBlocks = [7, 15, 23, 31], CrossFrameBlocks = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31], DimsReassemble = [256, 512, 1024, 1024],
        DimDecoder = 256,
    };

    /// <summary>Every size, smallest first.</summary>
    public static IReadOnlyList<DepthPro2Preset> All => [Small, Base, Large, HugePlus];

    /// <summary>A DINOv3 preset in the form Depth Pro 2 and its focal estimator embed it.</summary>
    public static Dinov2Preset TimmBackbone(Dinov2Preset dinov3) => dinov3 with
    {
        AttentionQkvBias = false, HasFinalNorm = false, RopeNormalizesByLongSide = true, RoundRopePeriodsToBf16 = false,
    };
}
