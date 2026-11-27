namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>A picture of the subject from a camera other than the front, for <see cref="HunyuanPaintPipeline"/> to bake onto the surfaces that camera sees.</summary>
/// <param name="Azimuth">The camera's azimuth in degrees at elevation 0, in <see cref="ThreeD.Paint.PaintCamera"/>'s convention (0 is the front).</param>
/// <param name="Rgb">The picture on white, RGB bytes, row 0 first.</param>
/// <param name="Alpha">Its subject mask, one byte per pixel.</param>
public sealed record PaintSideReference(double Azimuth, byte[] Rgb, byte[] Alpha, int Width, int Height);
