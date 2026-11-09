using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>Host reference for <see cref="IBackend.DepthwiseConvFrames"/>: taps summed in index order in F32; frames fan out.</summary>
public static class DepthwiseConvFramesReference
{
    /// <summary>Validates the operands of <see cref="IBackend.DepthwiseConvFrames"/>.</summary>
    public static void Validate(Tensor output, Tensor input, Tensor weight, int frames, int spatial, int channels, int taps, int pad)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(weight);
        if (frames < 1 || spatial < 1 || channels < 1 || taps < 1 || pad < 0 || pad >= taps)
            throw new ArgumentException($"DepthwiseConvFrames needs positive extents and 0 <= pad < taps, got {frames}x{spatial}x{channels}, "
                + $"taps={taps}, pad={pad}.");
        if (input.DType != DType.F32 || output.DType != DType.F32 || weight.DType != DType.F32)
            throw new ArgumentException("DepthwiseConvFrames operands must be F32.", nameof(input));
        long elements = (long)frames * spatial * channels;
        if (input.ElementCount != elements || output.ElementCount != elements)
            throw new ArgumentException($"DepthwiseConvFrames input {input.Shape} and output {output.Shape} must hold {elements} elements.", nameof(input));
        if (weight.ElementCount != (long)channels * taps)
            throw new ArgumentException($"DepthwiseConvFrames weight {weight.Shape} must hold {channels}x{taps} taps.", nameof(weight));
    }

    /// <summary>Host <see cref="IBackend.DepthwiseConvFrames"/>.</summary>
    public static unsafe void Run(Tensor output, Tensor input, Tensor weight, int frames, int spatial, int channels, int taps, int pad)
    {
        Validate(output, input, weight, frames, spatial, channels, taps, pad);
        nint o = (nint)output.DataPointer, i = (nint)input.DataPointer, w = (nint)weight.DataPointer;
        long plane = (long)spatial * channels;
        CpuParallel.For(frames, (long)frames * plane * taps, (o, i, w), (t, p) =>
        {
            float* outFrame = (float*)p.o + t * plane;
            float* x = (float*)p.i;
            float* taps0 = (float*)p.w;
            for (long e = 0; e < plane; e++)
            {
                int c = (int)(e % channels);
                float acc = 0f;
                for (int j = 0; j < taps; j++)
                {
                    int source = t + j - pad;
                    if (source < 0 || source >= frames) continue;
                    acc += taps0[(long)c * taps + j] * x[source * plane + e];
                }
                outFrame[e] = acc;
            }
        });
    }
}
