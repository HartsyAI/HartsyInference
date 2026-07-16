using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Vision.Annotators;

/// <summary>Lineart pre/post-processing matching comfyui_controlnet_aux's <c>LineartDetector</c>: RGB /255
/// in, Generator sketch out (white background, dark lines), then the detector's uint8 truncation and
/// <c>255 − x</c> inversion — the lineart ControlNets condition on white lines over a black background.
/// Host code by design: runs once per generation, outside the per-step hot path. The caller resizes the
/// source image; dimensions must be multiples of 4 (controlnet_aux snaps to multiples of 64).</summary>
public sealed class LineartPreprocessor
{
    private readonly LineartGenerator _model;

    public LineartPreprocessor(LineartGenerator model)
    {
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
    }

    /// <summary>Wrapped generator — exposed for weight preloading and variant identification.</summary>
    public LineartGenerator Model => _model;

    /// <summary>Converts interleaved RGB24 to the network input <c>[1,3,H,W]</c> in [0,1].</summary>
    public static Tensor Preprocess(ReadOnlySpan<byte> rgbPixels, int width, int height) =>
        ImageTensor.RgbToTensor01(rgbPixels, width, height);

    /// <summary>Runs the generator and returns the ControlNet conditioning map as a row-major
    /// <c>width·height</c> [0,1] buffer on the uint8 grid (black background, white lines).</summary>
    public float[] Process(IBackend backend, ReadOnlySpan<byte> rgbPixels, int width, int height)
    {
        using Tensor input = Preprocess(rgbPixels, width, height);
        Tensor line = _model.Forward(backend, input);
        float[] unit = PostprocessToUnit(line);
        line.Dispose();
        return unit;
    }

    /// <summary>Generator output <c>[1,1,H,W]</c> → inverted [0,1] conditioning buffer, replicating the
    /// Python pipeline exactly: <c>(line·255).clip(0,255).astype(uint8)</c> (truncation) then
    /// <c>255 − x</c>, scaled back to [0,1].</summary>
    public static unsafe float[] PostprocessToUnit(Tensor line)
    {
        if (line.Shape.Rank != 4 || line.Shape[0] != 1 || line.Shape[1] != 1)
            throw new ArgumentException($"line must be [1,1,H,W]; got {line.Shape}.", nameof(line));
        long count = line.ElementCount;
        float* src = (float*)line.DataPointer;
        float[] unit = new float[count];
        for (long i = 0; i < count; i++)
            unit[i] = (255 - HedPreprocessor.QuantizeU8(src[i])) * (1f / 255f);
        return unit;
    }

    /// <summary>Renders a processed unit map as interleaved-RGB24 grayscale.</summary>
    public static byte[] ToGrayscaleRgb24(ReadOnlySpan<float> unit) => ImageTensor.UnitGrayscaleToRgb24(unit);
}
