using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>The bidirectional depthwise temporal conv on SANA-WM keys, <c>conv(x) + conv(flip x).flip − w[3]·x</c> with a k=4 kernel.
/// That sum is one symmetric 7-tap filter <c>[w0 w1 w2 w3 w2 w1 w0]</c> over frames with zero padding, run as a depthwise conv.</summary>
internal static class SanaWmTemporalConv
{
    /// <summary>Taps of the stored kernel; the last multiplies the current frame.</summary>
    internal const int SourceTaps = 4;

    private const int Taps = 2 * SourceTaps - 1;

    /// <summary>Folds an F32 or BF16 <c>[C, 1, 4]</c> kernel into a host F32 depthwise kernel <c>[C, 1, 7, 1]</c>.</summary>
    internal static Tensor SymmetricKernel(Tensor conv, int channels, string name)
    {
        if (conv.Shape.Rank != 3 || conv.Shape[0] != channels || conv.Shape[1] != 1 || conv.Shape[2] != SourceTaps)
            throw new ArgumentException($"{name} must be [{channels}, 1, {SourceTaps}]; got {conv.Shape}.", name);
        List<Tensor> owned = [];
        try
        {
            ReadOnlySpan<float> source = SanaWmTensors.HostF32(conv, owned).AsReadOnlySpan<float>();
            Tensor kernel = new(new TensorShape(channels, 1, Taps, 1), DType.F32);
            Span<float> taps = kernel.AsSpan<float>();
            for (int c = 0; c < channels; c++)
            {
                for (int j = 0; j < Taps; j++)
                    taps[c * Taps + j] = source[c * SourceTaps + SourceTaps - 1 - Math.Abs(j - (SourceTaps - 1))];
            }
            return kernel;
        }
        finally
        {
            SanaWmTensors.DisposeAll(owned);
        }
    }

    /// <summary>Convolves token-major <c>[frames·spatial, C]</c> F32 along frames; returns a new <c>[frames·spatial, C]</c> tensor.</summary>
    internal static Tensor Apply(IBackend backend, Tensor input, Tensor kernel, int frames, int spatial, int channels)
    {
        int tokens = checked(frames * spatial);
        using SanaWmScratch scratch = new();
        Tensor channelFirst = scratch.New(1, channels, frames, spatial);
        backend.Transpose2D(channelFirst, input, tokens, channels);
        Tensor convolved = scratch.New(1, channels, frames, spatial);
        backend.Conv2dDepthwise(convolved, channelFirst, kernel, null, 1, 1, SourceTaps - 1, 0);
        scratch.Release(channelFirst);
        Tensor output = scratch.New(tokens, channels);
        backend.Transpose2D(output, convolved, channels, tokens);
        return scratch.Detach(output);
    }
}
