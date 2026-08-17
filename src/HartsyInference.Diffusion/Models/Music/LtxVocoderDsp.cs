using System.Collections.Generic;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>Shared DSP for the LTX-2 vocoder: deterministic low-pass kernels for the anti-aliasing and
/// band-extension resampling, plus the depthwise filter-broadcast and transposed-conv upsample helpers built on
/// them. The kernels mirror the reference <c>kaiser_sinc_filter1d</c> and the Hann-windowed sinc of
/// <c>UpSample1d</c>; the vocoder checkpoint also stores (identical) Kaiser buffers, but the Hann resampler filter
/// is non-persistent and must be computed regardless, so all filters are generated here for parity and
/// simplicity.</summary>
internal static unsafe class LtxVocoderDsp
{
    // Twin of NeuCodecEncoder.cs:631 + Audio/Io/Resampler.cs BesselI0 — Diffusion cannot reference Audio, and moving DSP into Core would be scope creep.
    /// <summary>Kaiser-windowed sinc low-pass, length <paramref name="kernelSize"/>, summing to 1. Matches the
    /// reference <c>kaiser_sinc_filter1d(cutoff, half_width, kernel_size)</c> (window-method FIR design: the Kaiser
    /// β is derived from the transition <c>delta_f = 4 · half_width</c> via the Kaiser formula).</summary>
    public static float[] KaiserSincFilter(double cutoff, double halfWidth, int kernelSize)
    {
        double deltaF = 4.0 * halfWidth;
        int halfSize = kernelSize / 2;
        double amplitude = 2.285 * (halfSize - 1) * Math.PI * deltaF + 7.95;
        double beta;
        if (amplitude > 50.0) beta = 0.1102 * (amplitude - 8.7);
        else if (amplitude >= 21.0) beta = 0.5842 * Math.Pow(amplitude - 21.0, 0.4) + 0.07886 * (amplitude - 21.0);
        else beta = 0.0;

        double[] window = KaiserWindow(kernelSize, beta);
        bool even = kernelSize % 2 == 0;
        float[] filter = new float[kernelSize];
        if (cutoff == 0.0) return filter;  // all zeros

        double sum = 0.0;
        for (int i = 0; i < kernelSize; i++)
        {
            double time = even ? (-halfSize + i) + 0.5 : i - halfSize;
            time *= 2.0 * cutoff;
            double sinc = time == 0.0 ? 1.0 : Math.Sin(Math.PI * time) / (Math.PI * time);
            double v = 2.0 * cutoff * window[i] * sinc;
            filter[i] = (float)v;
            sum += v;
        }
        if (sum != 0.0)
        {
            float inv = (float)(1.0 / sum);
            for (int i = 0; i < kernelSize; i++) filter[i] *= inv;
        }
        return filter;
    }

    /// <summary>Hann-windowed sinc low-pass used by the band-extension resampler (<c>UpSample1d</c> with
    /// <c>window_type="hann"</c>). Returns the filter plus the geometry (<paramref name="pad"/>,
    /// <paramref name="cropLeft"/>, <paramref name="cropRight"/>) the transposed-conv upsample needs.</summary>
    public static float[] HannSincFilter(int ratio, out int pad, out int cropLeft, out int cropRight)
    {
        const double rolloff = 0.99;
        const int lowpassFilterWidth = 6;
        int width = (int)Math.Ceiling(lowpassFilterWidth / rolloff);
        int kernelSize = 2 * width * ratio + 1;
        pad = width;
        cropLeft = 2 * width * ratio;
        cropRight = kernelSize - ratio;

        float[] filter = new float[kernelSize];
        for (int i = 0; i < kernelSize; i++)
        {
            double timeAxis = ((double)i / ratio - width) * rolloff;
            double clamped = Math.Clamp(timeAxis, -lowpassFilterWidth, lowpassFilterWidth);
            double window = Math.Pow(Math.Cos(clamped * Math.PI / lowpassFilterWidth / 2.0), 2.0);
            double sinc = timeAxis == 0.0 ? 1.0 : Math.Sin(Math.PI * timeAxis) / (Math.PI * timeAxis);
            filter[i] = (float)(sinc * window * rolloff / ratio);
        }
        return filter;
    }

    /// <summary>Broadcasts a shared 1-D lowpass filter into depthwise conv weights <c>[channels, 1, K]</c> (every
    /// channel gets the one kernel). Caller owns the returned tensor.</summary>
    public static Tensor DepthwiseFilter(float[] filter, int channels)
    {
        int kernel = filter.Length;
        Tensor t = new(new TensorShape(channels, 1, kernel), DType.F32);
        float* tp = (float*)t.DataPointer;
        for (int c = 0; c < channels; c++)
            for (int k = 0; k < kernel; k++)
                tp[(long)c * kernel + k] = filter[k];
        return t;
    }

    /// <summary>Sinc-interpolation upsample ×<paramref name="ratio"/> (reference <c>UpSample1d</c>): replicate-pad
    /// the time axis, depthwise transposed conv with the <c>[C, 1, K]</c> lowpass <paramref name="weight"/>, scale
    /// by the ratio, crop. The weight is borrowed, never disposed. Caller owns the returned tensor.</summary>
    public static Tensor DepthwiseUpsample(IBackend backend, Tensor x, Tensor weight, int ratio, int pad, int cropLeft, int cropRight)
    {
        int c = (int)x.Shape[1];
        int kernel = (int)weight.Shape[2];
        Tensor padded = LtxAudioDeviceOps.ReplicatePadTime(backend, x, pad, pad);
        int tp = (int)padded.Shape[2];
        int tOut = (tp - 1) * ratio + (kernel - 1) + 1;
        Tensor convt = new(new TensorShape(1, c, tOut), DType.F32);
        backend.ConvTranspose1d(convt, padded, weight, null, ratio, 0, 0, 1, c);
        padded.Dispose();
        backend.Scale(convt, convt, ratio);
        Tensor o = LtxAudioDeviceOps.CropTime(backend, convt, cropLeft, cropRight);
        convt.Dispose();
        return o;
    }

    /// <summary>Optional checkpoint tensor: the value when present, else null (biases and other optional weights).</summary>
    public static Tensor? OptionalWeight(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? t) ? t : null;

    /// <summary>Symmetric (non-periodic) Kaiser window of length <paramref name="n"/> with shape parameter
    /// <paramref name="beta"/>, matching <c>torch.kaiser_window(n, beta, periodic=False)</c>.</summary>
    private static double[] KaiserWindow(int n, double beta)
    {
        double[] w = new double[n];
        if (n == 1) { w[0] = 1.0; return w; }
        double denom = BesselI0(beta);
        double alpha = (n - 1) / 2.0;
        for (int i = 0; i < n; i++)
        {
            double r = (i - alpha) / alpha;
            double arg = beta * Math.Sqrt(Math.Max(0.0, 1.0 - r * r));
            w[i] = BesselI0(arg) / denom;
        }
        return w;
    }

    /// <summary>Modified Bessel function of the first kind, order 0, via its convergent power series.</summary>
    private static double BesselI0(double x)
    {
        double sum = 1.0;
        double term = 1.0;
        double halfX = x / 2.0;
        for (int k = 1; k < 64; k++)
        {
            term *= (halfX / k) * (halfX / k);
            sum += term;
            if (term < sum * 1e-16) break;
        }
        return sum;
    }
}
