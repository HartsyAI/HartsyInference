using HartsyInference.ThreeD.Models.Sam3dObjects;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>One SAM 3D Objects generation: an image, the mask of the object to lift and, optionally, a pointmap of the scene (MoGe estimates one when it is missing).</summary>
/// <param name="Rgb">Interleaved 8-bit RGB, <c>Height x Width x 3</c>.</param>
/// <param name="Mask">One byte per pixel, <c>Height x Width</c>; non-zero (255) marks the object, as the alpha channel of the upstream RGBA input.</param>
/// <param name="Width">Image width.</param>
/// <param name="Height">Image height.</param>
/// <param name="Pointmap">Planar <c>[3, h, w]</c> camera-space points in the PyTorch3D camera convention (what MoGe's output becomes after the upstream camera rotation); NaN or infinity marks an invalid point.
/// Null runs MoGe on the image (<c>moge_vitl.safetensors</c> beside the other checkpoints).</param>
/// <param name="Seed">Random seed; null picks one.</param>
/// <param name="StructureSteps">Euler steps of the structure flow (upstream 25).</param>
/// <param name="SlatSteps">Euler steps of the SLAT flow (upstream 25).</param>
/// <param name="Mesh">Decode the SLAT with the mesh decoder and extract a triangle mesh (FlexiCubes at 256) instead of Gaussian splats; needs <c>slat_decoder_mesh.safetensors</c>. The result then holds <see cref="Sam3dObjectsResult.Mesh"/> and no splats.</param>
/// <param name="MogeResolutionLevel">0 to 9, MoGe's token budget between its minimum and maximum (upstream 9); read only when the pointmap is estimated.</param>
public sealed record Sam3dObjectsRequest(
    byte[] Rgb,
    byte[] Mask,
    int Width,
    int Height,
    SamPointmapInput? Pointmap = null,
    int? Seed = null,
    int StructureSteps = 25,
    int SlatSteps = 25,
    int MogeResolutionLevel = 9,
    bool Mesh = false);
