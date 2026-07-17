namespace HartsyInference.Diffusion.Adapters;

/// <summary>Control-type ids for union SDXL UNet ControlNets (xinsir/controlnet-union-sdxl-1.0). The integer
/// values are row indices into the checkpoint's <c>task_embedding</c> table and positions in the multi-hot
/// <c>control_type</c> vector — they must match the training mapping exactly (xinsir6/ControlNetPlus README:
/// 0 openpose, 1 depth, 2 thick line (scribble/hed/softedge/ted), 3 thin line (canny/mlsd/lineart/anime
/// lineart), 4 normal, 5 segment; the ProMax revision appends 6 tile and 7 repaint). The standard union
/// checkpoint has a 6- or 8-row table depending on revision — <see cref="ControlNetConfig.UnionControlTypeCount"/>
/// carries the actual row count so out-of-range ids fail fast.</summary>
public enum SdxlUnionControlType
{
    /// <summary>OpenPose skeleton.</summary>
    OpenPose = 0,

    /// <summary>Depth map.</summary>
    Depth = 1,

    /// <summary>Thick-line maps: scribble / HED / PIDI soft edge / TEED.</summary>
    SoftEdge = 2,

    /// <summary>Thin-line maps: canny / MLSD / lineart / anime lineart.</summary>
    Canny = 3,

    /// <summary>Normal map.</summary>
    Normal = 4,

    /// <summary>Semantic segmentation map.</summary>
    Segment = 5,

    /// <summary>Tile / deblur / super-resolution conditioning (ProMax revision only).</summary>
    Tile = 6,

    /// <summary>Inpaint / outpaint repaint conditioning (ProMax revision only).</summary>
    Repaint = 7,
}
