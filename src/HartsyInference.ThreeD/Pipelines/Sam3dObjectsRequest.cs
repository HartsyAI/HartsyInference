using HartsyInference.ThreeD.Models.Sam3dObjects;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>One SAM 3D Objects generation: an image, the mask of the object to lift and a pointmap of the scene.</summary>
/// <param name="Rgb">Interleaved 8-bit RGB, <c>Height x Width x 3</c>.</param>
/// <param name="Mask">One byte per pixel, <c>Height x Width</c>; non-zero (255) marks the object, as the alpha channel of the upstream RGBA input.</param>
/// <param name="Width">Image width.</param>
/// <param name="Height">Image height.</param>
/// <param name="Pointmap">Planar <c>[3, h, w]</c> camera-space points in the PyTorch3D camera convention (what MoGe's output becomes after the upstream camera rotation); NaN or infinity marks an invalid point.</param>
/// <param name="Seed">Random seed; null picks one.</param>
/// <param name="StructureSteps">Euler steps of the structure flow (upstream 25).</param>
/// <param name="SlatSteps">Euler steps of the SLAT flow (upstream 25).</param>
public sealed record Sam3dObjectsRequest(
    byte[] Rgb,
    byte[] Mask,
    int Width,
    int Height,
    SamPointmapInput Pointmap,
    int? Seed = null,
    int StructureSteps = 25,
    int SlatSteps = 25);
