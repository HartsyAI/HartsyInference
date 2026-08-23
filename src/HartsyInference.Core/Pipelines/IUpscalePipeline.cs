namespace HartsyInference.Core.Pipelines;

/// <summary>Image super-resolution pipeline (ESRGAN / Real-ESRGAN family). Takes an RGB image and returns a higher-resolution RGB image enlarged by the model's scale factor.</summary>
public interface IUpscalePipeline : IVisionPipeline
{
    /// <summary>The integer upscale factor (e.g. 4 for Real-ESRGAN x4plus).</summary>
    int ScaleFactor { get; }

    /// <summary>Upscales an RGB image. Input/output are HWC byte buffers in [0,255].</summary>
    /// <param name="rgbData">Source pixels, length <c>width*height*3</c>.</param>
    /// <param name="width">Source width.</param>
    /// <param name="height">Source height.</param>
    /// <returns>(upscaled RGB bytes, outWidth = width*scale, outHeight = height*scale).</returns>
    (byte[] rgbData, int width, int height) Upscale(ReadOnlySpan<byte> rgbData, int width, int height);
}
