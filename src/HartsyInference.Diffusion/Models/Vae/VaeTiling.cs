using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Shared tile extraction / padding / cropping / linear-ramp blending / concatenation
/// helpers for the tiled VAE wrappers. Both <see cref="VaeTiledDecoder"/> (tiles + blends in pixel
/// space) and <see cref="VaeTiledEncoder"/> (tiles in pixel space, blends in latent space) use these.
/// Every method operates generically on F32 [B, C, H, W] tensors with the channel count passed in,
/// so the same code serves 3-channel RGB and N-channel latents. Also reused by the Vision upscaler
/// (RRDBNet tiling), where the spatial scale factor is the upscale ratio rather than the VAE's 8x.</summary>
public static unsafe class VaeTiling
{
    /// <summary>Extracts a tile [B, C, tileH, tileW] from <paramref name="source"/> at position (startH, startW).</summary>
    public static Tensor ExtractTile(Tensor source, int batch, int channels, int startH, int startW, int tileH, int tileW)
    {
        int srcH = (int)source.Shape[2];
        int srcW = (int)source.Shape[3];
        TensorShape tileShape = new TensorShape(batch, channels, tileH, tileW);
        Tensor tile = new Tensor(tileShape, DType.F32);

        float* srcPtr = (float*)source.DataPointer;
        float* dstPtr = (float*)tile.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < tileH; h++)
                {
                    int srcOffset = ((b * channels + c) * srcH + (startH + h)) * srcW + startW;
                    int dstOffset = ((b * channels + c) * tileH + h) * tileW;

                    Buffer.MemoryCopy(
                        srcPtr + srcOffset,
                        dstPtr + dstOffset,
                        tileW * sizeof(float),
                        tileW * sizeof(float));
                }
            }
        }

        return tile;
    }

    /// <summary>Pads a tile to targetH x targetW with zeros (top-left aligned).</summary>
    public static Tensor PadTile(IBackend backend, Tensor tile, int batch, int channels, int targetH, int targetW)
    {
        TensorShape paddedShape = new TensorShape(batch, channels, targetH, targetW);
        Tensor padded = new Tensor(paddedShape, DType.F32);
        backend.Fill(padded, 0f);

        int tileH = (int)tile.Shape[2];
        int tileW = (int)tile.Shape[3];
        float* srcPtr = (float*)tile.DataPointer;
        float* dstPtr = (float*)padded.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < tileH; h++)
                {
                    int srcOff = ((b * channels + c) * tileH + h) * tileW;
                    int dstOff = ((b * channels + c) * targetH + h) * targetW;
                    Buffer.MemoryCopy(srcPtr + srcOff, dstPtr + dstOff, tileW * sizeof(float), tileW * sizeof(float));
                }
            }
        }

        return padded;
    }

    /// <summary>Crops a tile to cropH x cropW (top-left aligned).</summary>
    public static Tensor CropTile(Tensor tile, int batch, int channels, int cropH, int cropW)
    {
        TensorShape croppedShape = new TensorShape(batch, channels, cropH, cropW);
        Tensor cropped = new Tensor(croppedShape, DType.F32);

        float* srcPtr = (float*)tile.DataPointer;
        float* dstPtr = (float*)cropped.DataPointer;
        int srcW = (int)tile.Shape[3];
        int srcH = (int)tile.Shape[2];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < cropH; h++)
                {
                    int srcOff = ((b * channels + c) * srcH + h) * srcW;
                    int dstOff = ((b * channels + c) * cropH + h) * cropW;
                    Buffer.MemoryCopy(srcPtr + srcOff, dstPtr + dstOff, cropW * sizeof(float), cropW * sizeof(float));
                }
            }
        }

        return cropped;
    }

    /// <summary>Blends the overlap region between left and right tiles horizontally with a linear ramp. Modifies the right tile in-place.</summary>
    public static void BlendHorizontal(Tensor left, Tensor right, int blendExtent)
    {
        int batch = (int)right.Shape[0];
        int channels = (int)right.Shape[1];
        int height = (int)right.Shape[2];
        int leftW = (int)left.Shape[3];
        int rightW = (int)right.Shape[3];

        int actualBlend = Math.Min(Math.Min(leftW, rightW), blendExtent);

        float* leftPtr = (float*)left.DataPointer;
        float* rightPtr = (float*)right.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int x = 0; x < actualBlend; x++)
                    {
                        float weight = (float)x / actualBlend;
                        int leftIdx = ((b * channels + c) * (int)left.Shape[2] + h) * leftW + (leftW - actualBlend + x);
                        int rightIdx = ((b * channels + c) * height + h) * rightW + x;

                        rightPtr[rightIdx] = leftPtr[leftIdx] * (1.0f - weight) + rightPtr[rightIdx] * weight;
                    }
                }
            }
        }
    }

    /// <summary>Blends the overlap region between top and bottom tiles vertically with a linear ramp. Modifies the bottom tile in-place.</summary>
    public static void BlendVertical(Tensor top, Tensor bottom, int blendExtent)
    {
        int batch = (int)bottom.Shape[0];
        int channels = (int)bottom.Shape[1];
        int topH = (int)top.Shape[2];
        int bottomH = (int)bottom.Shape[2];
        int width = (int)bottom.Shape[3];
        int topW = (int)top.Shape[3];

        int actualBlend = Math.Min(Math.Min(topH, bottomH), blendExtent);

        float* topPtr = (float*)top.DataPointer;
        float* bottomPtr = (float*)bottom.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int y = 0; y < actualBlend; y++)
                {
                    float weight = (float)y / actualBlend;
                    int topRow = ((b * channels + c) * topH + (topH - actualBlend + y)) * topW;
                    int bottomRow = ((b * channels + c) * bottomH + y) * width;

                    for (int x = 0; x < Math.Min(width, topW); x++)
                    {
                        bottomPtr[bottomRow + x] = topPtr[topRow + x] * (1.0f - weight) + bottomPtr[bottomRow + x] * weight;
                    }
                }
            }
        }
    }

    /// <summary>Concatenates tiles horizontally, cropping all but the last to <paramref name="rowLimit"/> width.</summary>
    public static Tensor ConcatHorizontal(Tensor[] tiles, int rowLimit, int batch)
    {
        int channels = (int)tiles[0].Shape[1];
        int height = (int)tiles[0].Shape[2];

        int totalWidth = 0;
        for (int i = 0; i < tiles.Length; i++)
        {
            int tileW = (int)tiles[i].Shape[3];
            totalWidth += (i < tiles.Length - 1) ? Math.Min(rowLimit, tileW) : tileW;
        }

        TensorShape resultShape = new TensorShape(batch, channels, height, totalWidth);
        Tensor result = new Tensor(resultShape, DType.F32);
        float* dstPtr = (float*)result.DataPointer;

        int wOffset = 0;
        for (int t = 0; t < tiles.Length; t++)
        {
            int tileW = (int)tiles[t].Shape[3];
            int copyW = (t < tiles.Length - 1) ? Math.Min(rowLimit, tileW) : tileW;
            float* srcPtr = (float*)tiles[t].DataPointer;

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        int srcOff = ((b * channels + c) * height + h) * tileW;
                        int dstOff = ((b * channels + c) * height + h) * totalWidth + wOffset;
                        Buffer.MemoryCopy(srcPtr + srcOff, dstPtr + dstOff, copyW * sizeof(float), copyW * sizeof(float));
                    }
                }
            }

            wOffset += copyW;
        }

        return result;
    }

    /// <summary>Concatenates rows vertically, cropping all but the last to <paramref name="rowLimit"/> height.</summary>
    public static Tensor ConcatVertical(Tensor[] rows, int rowLimit, int batch)
    {
        int channels = (int)rows[0].Shape[1];
        int width = (int)rows[0].Shape[3];

        int totalHeight = 0;
        for (int i = 0; i < rows.Length; i++)
        {
            int rowH = (int)rows[i].Shape[2];
            totalHeight += (i < rows.Length - 1) ? Math.Min(rowLimit, rowH) : rowH;
        }

        TensorShape resultShape = new TensorShape(batch, channels, totalHeight, width);
        Tensor result = new Tensor(resultShape, DType.F32);
        float* dstPtr = (float*)result.DataPointer;

        int hOffset = 0;
        for (int r = 0; r < rows.Length; r++)
        {
            int rowH = (int)rows[r].Shape[2];
            int rowW = (int)rows[r].Shape[3];
            int copyH = (r < rows.Length - 1) ? Math.Min(rowLimit, rowH) : rowH;
            float* srcPtr = (float*)rows[r].DataPointer;

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < copyH; h++)
                    {
                        int srcOff = ((b * channels + c) * rowH + h) * rowW;
                        int dstOff = ((b * channels + c) * totalHeight + (hOffset + h)) * width;
                        int copyBytes = Math.Min(width, rowW) * sizeof(float);
                        Buffer.MemoryCopy(srcPtr + srcOff, dstPtr + dstOff, copyBytes, copyBytes);
                    }
                }
            }

            hOffset += copyH;
        }

        return result;
    }
}
