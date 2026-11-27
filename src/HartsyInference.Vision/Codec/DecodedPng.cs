namespace HartsyInference.Vision.Codec;

/// <summary>A decoded PNG: RGB24 and, when the file has one, an 8-bit alpha plane (null when opaque), both row-major.
/// <see cref="ExifOrientation"/> is the file's EXIF orientation; the pixels are already oriented when <see cref="OrientationApplied"/> is set.</summary>
public sealed record DecodedPng(byte[] Rgb, byte[]? Alpha, int Width, int Height, int ExifOrientation, bool OrientationApplied);
