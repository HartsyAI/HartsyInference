using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Vision.Annotators;

/// <summary>HED soft-edge pre/post-processing matching comfyui_controlnet_aux's <c>HEDdetector</c>: raw
/// [0,255] RGB in, fused sigmoid edge map out, quantized through the same uint8 truncation the Python
/// pipeline applies, so conditioning bytes match bit-for-bit. Also implements the <c>scribble_hed</c>
/// post-process (Gaussian blur → 4-direction line NMS → threshold → blur → binarize) and the <c>safe</c>
/// step quantization. Host code by design: runs once per generation, outside the per-step hot path.
/// The caller resizes the source image (controlnet_aux snaps to multiples of 64; any size works here).</summary>
public sealed class HedPreprocessor
{
    private readonly HedModel _model;

    public HedPreprocessor(HedModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
    }

    /// <summary>Wrapped model — exposed for weight preloading.</summary>
    public HedModel Model => _model;

    /// <summary>Converts interleaved RGB24 to the network input <c>[1,3,H,W]</c> (raw [0,255] floats).</summary>
    public static Tensor Preprocess(ReadOnlySpan<byte> rgbPixels, int width, int height) =>
        ImageTensor.RgbToTensor255(rgbPixels, width, height);

    /// <summary>Runs the detector and returns the soft edge map as a row-major <c>width·height</c> [0,1]
    /// buffer on the uint8 grid (black background, white edges — the softedge_hed conditioning form).</summary>
    public float[] Process(IBackend backend, ReadOnlySpan<byte> rgbPixels, int width, int height, bool safe = false)
    {
        using Tensor input = Preprocess(rgbPixels, width, height);
        Tensor edge = _model.Forward(backend, input);
        float[] unit = PostprocessToUnit(edge, safe);
        edge.Dispose();
        return unit;
    }

    /// <summary>Runs the detector and returns the binarized scribble_hed map ({0,1} values).</summary>
    public float[] ProcessScribble(IBackend backend, ReadOnlySpan<byte> rgbPixels, int width, int height)
    {
        float[] soft = Process(backend, rgbPixels, width, height);
        byte[] gray = new byte[soft.Length];
        for (int i = 0; i < soft.Length; i++) gray[i] = (byte)MathF.Round(soft[i] * 255f);
        byte[] scribble = ScribbleFromSoftEdge(gray, width, height);
        float[] unit = new float[scribble.Length];
        for (int i = 0; i < scribble.Length; i++) unit[i] = scribble[i] * (1f / 255f);
        return unit;
    }

    /// <summary>Fused edge <c>[1,1,H,W]</c> → [0,1] buffer quantized exactly like the Python pipeline's
    /// <c>(edge·255).clip(0,255).astype(uint8)</c> (truncation, not rounding). <paramref name="safe"/>
    /// applies controlnet_aux <c>safe_step</c> (<c>floor(x·3)/2</c>) before quantization.</summary>
    public static unsafe float[] PostprocessToUnit(Tensor edge, bool safe = false)
    {
        if (edge.Shape.Rank != 4 || edge.Shape[0] != 1 || edge.Shape[1] != 1)
            throw new ArgumentException($"edge must be [1,1,H,W]; got {edge.Shape}.", nameof(edge));
        long count = edge.ElementCount;
        float* src = (float*)edge.DataPointer;
        float[] unit = new float[count];
        for (long i = 0; i < count; i++)
        {
            float x = src[i];
            if (safe) x = SafeStep(x);
            unit[i] = QuantizeU8(x) * (1f / 255f);
        }
        return unit;
    }

    /// <summary>controlnet_aux <c>safe_step</c> with step=2: <c>float(int(x·3)) / 2</c>.</summary>
    public static float SafeStep(float x, int step = 2) => (int)(x * (step + 1)) / (float)step;

    /// <summary><c>(x·255).clip(0,255).astype(uint8)</c> — numpy truncation semantics.</summary>
    public static byte QuantizeU8(float x) => (byte)Math.Clamp((int)(x * 255f), 0, 255);

    /// <summary>Renders a processed unit map as interleaved-RGB24 grayscale.</summary>
    public static byte[] ToGrayscaleRgb24(ReadOnlySpan<float> unit) => ImageTensor.UnitGrayscaleToRgb24(unit);

    /// <summary>scribble_hed post-process on the quantized u8 soft edge map, matching controlnet_aux:
    /// <c>nms(map, 127, 3.0)</c> (Gaussian σ=3 blur, keep pixels that are maxima along any of the 4 line
    /// directions, threshold &gt; 127), then Gaussian σ=3 blur and binarize at &gt; 4. Returns 0/255 bytes.</summary>
    public static byte[] ScribbleFromSoftEdge(ReadOnlySpan<byte> gray, int width, int height)
    {
        if (gray.Length != width * height)
            throw new ArgumentException($"gray length {gray.Length} != {width}x{height}.", nameof(gray));

        float[] src = new float[gray.Length];
        for (int i = 0; i < gray.Length; i++) src[i] = gray[i];
        // cv2 auto kernel size for CV_32F input: round(sigma·4·2 + 1) | 1 = 25 at σ=3.
        float[] blurred = GaussianBlur(src, width, height, 3.0, 25);

        byte[] nms = new byte[gray.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float v = blurred[y * width + x];
                bool isMax =
                    IsLineMax(blurred, width, height, x, y, 1, 0) ||
                    IsLineMax(blurred, width, height, x, y, 0, 1) ||
                    IsLineMax(blurred, width, height, x, y, 1, 1) ||
                    IsLineMax(blurred, width, height, x, y, 1, -1);
                nms[y * width + x] = isMax && v > 127f ? (byte)255 : (byte)0;
            }
        }

        // cv2 auto kernel size for CV_8U input: round(sigma·3·2 + 1) | 1 = 19 at σ=3.
        float[] nmsF = new float[nms.Length];
        for (int i = 0; i < nms.Length; i++) nmsF[i] = nms[i];
        float[] blurred2 = GaussianBlur(nmsF, width, height, 3.0, 19);

        byte[] result = new byte[nms.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = MathF.Round(blurred2[i]) > 4f ? (byte)255 : (byte)0;
        return result;
    }

    /// <summary>Separable Gaussian blur with cv2 semantics: <c>getGaussianKernel(ksize, sigma)</c> taps and
    /// BORDER_REFLECT_101 edge handling.</summary>
    public static float[] GaussianBlur(ReadOnlySpan<float> src, int width, int height, double sigma, int ksize)
    {
        if (ksize % 2 == 0 || ksize <= 0)
            throw new ArgumentException($"Kernel size must be odd and positive; got {ksize}.", nameof(ksize));
        double[] kernel = GaussianKernel(ksize, sigma);
        int r = ksize / 2;

        float[] tmp = new float[src.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double acc = 0;
                for (int k = -r; k <= r; k++)
                    acc += kernel[k + r] * src[y * width + Reflect101(x + k, width)];
                tmp[y * width + x] = (float)acc;
            }
        }
        float[] outp = new float[src.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double acc = 0;
                for (int k = -r; k <= r; k++)
                    acc += kernel[k + r] * tmp[Reflect101(y + k, height) * width + x];
                outp[y * width + x] = (float)acc;
            }
        }
        return outp;
    }

    /// <summary>Normalized Gaussian taps, matching cv2 <c>getGaussianKernel</c> for σ &gt; 0.</summary>
    public static double[] GaussianKernel(int ksize, double sigma)
    {
        double[] kernel = new double[ksize];
        double center = (ksize - 1) / 2.0;
        double sum = 0;
        for (int i = 0; i < ksize; i++)
        {
            double d = i - center;
            kernel[i] = Math.Exp(-d * d / (2 * sigma * sigma));
            sum += kernel[i];
        }
        for (int i = 0; i < ksize; i++) kernel[i] /= sum;
        return kernel;
    }

    /// <summary>True when the pixel is ≥ both its neighbors along the (dx, dy) line direction —
    /// cv2 dilate-with-line-kernel equality, out-of-bounds neighbors ignored.</summary>
    private static bool IsLineMax(float[] img, int width, int height, int x, int y, int dx, int dy)
    {
        float v = img[y * width + x];
        int xa = x - dx, ya = y - dy, xb = x + dx, yb = y + dy;
        if (xa >= 0 && xa < width && ya >= 0 && ya < height && img[ya * width + xa] > v) return false;
        if (xb >= 0 && xb < width && yb >= 0 && yb < height && img[yb * width + xb] > v) return false;
        return true;
    }

    private static int Reflect101(int i, int n)
    {
        if (n == 1) return 0;
        while (i < 0 || i >= n)
        {
            if (i < 0) i = -i;
            if (i >= n) i = 2 * (n - 1) - i;
        }
        return i;
    }
}
