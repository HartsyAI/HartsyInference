using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

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
        _patchSize = patchSize;
        _outChannels = outChannels;
    }

    /// <summary>Rearranges patch tokens to spatial image. Input: [B, gridH*gridW, patchSize^2 * outChannels]. Output: [B, outChannels, gridH*patchSize, gridW*patchSize].</summary>
    public Tensor Forward(Tensor input, int batch, int gridH, int gridW)
    {
        int height = gridH * _patchSize;
        int width = gridW * _patchSize;
        TensorShape outShape = new TensorShape(batch, _outChannels, height, width);
        Tensor output = new Tensor(outShape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        int patchDim = _patchSize * _patchSize * _outChannels;

        // Rearrange: [B, gridH, gridW, patchH, patchW, C] → [B, C, gridH*patchH, gridW*patchW]
        // Input layout per token: patchH * patchW * C contiguous (spatial-first, channels-last within each patch)
        for (int b = 0; b < batch; b++)
        {
            for (int gy = 0; gy < gridH; gy++)
            {
                for (int gx = 0; gx < gridW; gx++)
                {
                    int patchIdx = gy * gridW + gx;
                    int inBase = (b * gridH * gridW + patchIdx) * patchDim;

                    for (int py = 0; py < _patchSize; py++)
                    {
                        for (int px = 0; px < _patchSize; px++)
                        {
                            int outY = gy * _patchSize + py;
                            int outX = gx * _patchSize + px;
                            int patchPixel = py * _patchSize + px;

                            for (int c = 0; c < _outChannels; c++)
                            {
                                int srcIdx = inBase + patchPixel * _outChannels + c;
                                int dstIdx = ((b * _outChannels + c) * height + outY) * width + outX;
                                outPtr[dstIdx] = inPtr[srcIdx];
                            }
                        }
                    }
                }
            }
        }

        return output;
    }
}
