using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Converts a sequence of patch tokens back to spatial layout. Reverses PatchEmbed by rearranging [B, numPatches, patchSize^2 * channels] → [B, channels, H, W].</summary>
public sealed unsafe class Unpatchify
{
    private readonly int _patchSize;
    private readonly int _outChannels;

    /// <summary>Creates an unpatchify layer.</summary>
    /// <param name="patchSize">Patch size (2 for SD3).</param>
    /// <param name="outChannels">Output latent channels (16 for SD3).</param>
    public Unpatchify(int patchSize, int outChannels)
    {
        if (patchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(patchSize), patchSize, "Patch size must be positive.");
        if (outChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(outChannels), outChannels, "Output channels must be positive.");
        _patchSize = patchSize;
        _outChannels = outChannels;
    }

    /// <summary>Rearranges patch tokens to spatial image. Input: [B, gridH*gridW, patchSize^2 * outChannels]. Output: [B, outChannels, gridH*patchSize, gridW*patchSize].</summary>
    public Tensor Forward(Tensor input, int batch, int gridH, int gridW)
    {
        (int height, int width) = ValidateInput(input, batch, gridH, gridW);
        TensorShape outShape = new TensorShape(batch, _outChannels, height, width);
        Tensor output = new Tensor(outShape, DType.F32);
        try
        {
            float* inPtr = (float*)input.DataPointer;
            float* outPtr = (float*)output.DataPointer;
            long patchDim = checked((long)_patchSize * _patchSize * _outChannels);

            // Rearrange: [B, gridH, gridW, patchH, patchW, C] → [B, C, gridH*patchH, gridW*patchW]
            // Input layout per token: patchH * patchW * C contiguous (spatial-first, channels-last within each patch)
            for (int b = 0; b < batch; b++)
            {
                for (int gy = 0; gy < gridH; gy++)
                {
                    for (int gx = 0; gx < gridW; gx++)
                    {
                        long patchIdx = (long)gy * gridW + gx;
                        long inBase = ((long)b * gridH * gridW + patchIdx) * patchDim;

                        for (int py = 0; py < _patchSize; py++)
                        {
                            for (int px = 0; px < _patchSize; px++)
                            {
                                int outY = gy * _patchSize + py;
                                int outX = gx * _patchSize + px;
                                long patchPixel = (long)py * _patchSize + px;

                                for (int c = 0; c < _outChannels; c++)
                                {
                                    long srcIdx = inBase + patchPixel * _outChannels + c;
                                    long dstIdx = (((long)b * _outChannels + c) * height + outY) * width + outX;
                                    outPtr[dstIdx] = inPtr[srcIdx];
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
        catch
        {
            DisposeBestEffort(output, "host output after failure");
            throw;
        }
    }

    /// <summary>Backend-resident rearrangement for arbitrary positive batch size. Input patch-token order is
    /// <c>[patchH, patchW, channel]</c>, matching SD3/AuraFlow's projection layout.</summary>
    public Tensor Forward(IBackend backend, Tensor input, int batch, int gridH, int gridW)
    {
        ArgumentNullException.ThrowIfNull(backend);
        (int height, int width) = ValidateInput(input, batch, gridH, gridW);

        Tensor output = new(new TensorShape(batch, _outChannels, height, width), DType.F32);
        try
        {
            // SD3 proj_out tokens are [patchH, patchW, channel] within each token.
            backend.UnpatchifyTokens(output, input, _outChannels, gridH, gridW, _patchSize,
                innerChannelFastest: true);
            return output;
        }
        catch
        {
            DisposeBestEffort(output, "backend output after failure");
            throw;
        }
    }

    private (int Height, int Width) ValidateInput(Tensor input, int batch, int gridH, int gridW)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (batch <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), batch, "Batch size must be positive.");
        if (gridH <= 0)
            throw new ArgumentOutOfRangeException(nameof(gridH), gridH, "Packed-grid height must be positive.");
        if (gridW <= 0)
            throw new ArgumentOutOfRangeException(nameof(gridW), gridW, "Packed-grid width must be positive.");
        if (input.DType != DType.F32)
            throw new NotSupportedException($"SD3 unpatchify requires F32 patch tokens; got {input.DType}.");

        int height;
        int width;
        long sequenceLength;
        long patchDim;
        try
        {
            height = checked(gridH * _patchSize);
            width = checked(gridW * _patchSize);
            sequenceLength = checked((long)gridH * gridW);
            patchDim = checked((long)_patchSize * _patchSize * _outChannels);
            _ = checked((long)batch * _outChannels * height * width);
        }
        catch (OverflowException error)
        {
            throw new ArgumentException(
                "SD3 unpatchify geometry exceeds its supported indexing range.", nameof(input), error);
        }

        TensorShape expected = new(batch, sequenceLength, patchDim);
        if (input.Shape != expected)
            throw new ArgumentException($"SD3 unpatchify input must have shape {expected}; got {input.Shape}.", nameof(input));
        return (height, width);
    }

    private static void DisposeBestEffort(Tensor tensor, string description)
    {
        try { tensor.Dispose(); }
        catch (Exception error) { Logs.Error($"Failed to release SD3 unpatchify {description}.", error); }
    }
}
