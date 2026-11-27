namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>What <see cref="Trellis2ImagePreprocessor"/> made of an image: the crop the conditioner resizes, and the cutout it was cropped from.</summary>
/// <param name="Crop">The premultiplied RGB square around the foreground.</param>
/// <param name="Rgb">The image the crop was taken from, shrunk to the preprocessor's longest edge when it was larger; interleaved RGB24.</param>
/// <param name="Alpha">The foreground alpha of <paramref name="Rgb"/>: the image's own, or the matte.</param>
/// <param name="Width">Width of <paramref name="Rgb"/> and <paramref name="Alpha"/>.</param>
/// <param name="Height">Height of <paramref name="Rgb"/> and <paramref name="Alpha"/>.</param>
/// <param name="Matted">True when <paramref name="Alpha"/> is a matte the matter produced.</param>
public readonly record struct Trellis2PreparedCutout(Trellis2PreparedImage Crop, byte[] Rgb, byte[] Alpha, int Width, int Height, bool Matted);
