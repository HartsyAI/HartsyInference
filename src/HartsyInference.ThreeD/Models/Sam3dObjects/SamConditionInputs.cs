namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// The preprocessed inputs of one SAM 3D condition, each already at the size the upstream preprocessor produces (<c>Resize(518)</c> after the pad to a square):
/// the <c>cropped</c> branch is the object's bounding box scaled by 1.2, the <c>full</c> branch the whole frame. Images are planar RGB <c>[3, 518, 518]</c> in 0 to 1,
/// masks <c>[518, 518]</c> in 0 to 1, pointmaps planar <c>[3, h, w]</c> (NaN or infinity marks an invalid point; the structure stage resizes them to 256 itself).
/// The SLAT stage reads only the images and masks.
/// </summary>
/// <param name="CroppedImage"><c>image</c> in the YAML.</param>
/// <param name="CroppedMask"><c>mask</c>.</param>
/// <param name="FullImage"><c>rgb_image</c>.</param>
/// <param name="FullMask"><c>rgb_image_mask</c>.</param>
/// <param name="CroppedPointmap"><c>pointmap</c>; null for the SLAT stage.</param>
/// <param name="FullPointmap"><c>rgb_pointmap</c>; null for the SLAT stage.</param>
public sealed record SamConditionInputs(
    float[] CroppedImage,
    float[] CroppedMask,
    float[] FullImage,
    float[] FullMask,
    SamPointmapInput? CroppedPointmap = null,
    SamPointmapInput? FullPointmap = null);

/// <summary>A planar <c>[3, Height, Width]</c> pointmap.</summary>
public sealed record SamPointmapInput(float[] Planar, int Height, int Width);
