using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Tiled VAE decoder that splits latents into overlapping tiles, decodes each independently, and blends with linear interpolation. Keeps VRAM usage constant regardless of output image size.</summary>
public sealed class VaeTiledDecoder
{
    private readonly VaeDecoder _decoder;
    private readonly float _tileOverlapFactor;
    private const int SpatialCompressionFactor = 8;

    /// <summary>Creates a tiled decoder wrapping the given VaeDecoder.</summary>
    /// <param name="decoder">The underlying VAE decoder to use for individual tiles.</param>
    /// <param name="tileOverlapFactor">Fraction of tile that overlaps with neighbors. Default: 0.25.</param>
    public VaeTiledDecoder(VaeDecoder decoder, float tileOverlapFactor = 0.25f)
    {
        _decoder = decoder;
        _tileOverlapFactor = tileOverlapFactor;
    }

    /// <summary>Decodes a latent tensor using tiled decoding. Input: [B, C, H, W] (latent space). Output: [B, 3, H*8, W*8] (pixel space).</summary>
    public unsafe Tensor Decode(IBackend backend, Tensor latent)
    {
        VaeConfig config = _decoder.Config;
        int batch = (int)latent.Shape[0];
        int latentCh = (int)latent.Shape[1];
        int latentH = (int)latent.Shape[2];
        int latentW = (int)latent.Shape[3];

        // Tile size in latent space
        int tileLatentSize = config.SampleSize / (int)Math.Pow(2, config.BlockOutChannels.Length - 1);
        int overlapSize = (int)(tileLatentSize * (1.0f - _tileOverlapFactor));
        int blendExtent = (int)(tileLatentSize * _tileOverlapFactor);
        int rowLimit = tileLatentSize - blendExtent;

        // If the latent fits in a single tile, just decode directly
        if (latentH <= tileLatentSize && latentW <= tileLatentSize)
        {
            return _decoder.Decode(backend, latent);
        }

        // Pixel-space blend extent (latent blend * compression factor)
        int pixelBlendExtent = blendExtent * SpatialCompressionFactor;
        int pixelRowLimit = rowLimit * SpatialCompressionFactor;
        int pixelTileSize = tileLatentSize * SpatialCompressionFactor;

        // Extract, decode, and store tiles in a grid
        int numRows = 0;
        for (int i = 0; i < latentH; i += overlapSize) numRows++;

        int numCols = 0;
        for (int j = 0; j < latentW; j += overlapSize) numCols++;

        Tensor[][] tileGrid = new Tensor[numRows][];

        int rowIdx = 0;
        for (int i = 0; i < latentH; i += overlapSize)
        {
            tileGrid[rowIdx] = new Tensor[numCols];
            int colIdx = 0;

            for (int j = 0; j < latentW; j += overlapSize)
            {
                // Extract tile from latent (clamp to bounds)
                int tileH = Math.Min(tileLatentSize, latentH - i);
                int tileW = Math.Min(tileLatentSize, latentW - j);

                Tensor tile = ExtractTile(latent, batch, latentCh, i, j, tileH, tileW);

                // Pad tile to full tile size if needed (at edges)
                Tensor paddedTile;
                if (tileH < tileLatentSize || tileW < tileLatentSize)
                {
                    paddedTile = PadTile(backend, tile, batch, latentCh, tileLatentSize, tileLatentSize);
                    tile.Dispose();
                }
                else
                {
                    paddedTile = tile;
                }

                // Decode this tile
                Tensor decoded = _decoder.Decode(backend, paddedTile);
                paddedTile.Dispose();

                // Crop decoded tile if we padded it
                int decodedH = tileH * SpatialCompressionFactor;
                int decodedW = tileW * SpatialCompressionFactor;
                if (decodedH < pixelTileSize || decodedW < pixelTileSize)
                {
                    Tensor cropped = CropTile(decoded, batch, 3, decodedH, decodedW);
                    decoded.Dispose();
                    decoded = cropped;
                }

                tileGrid[rowIdx][colIdx] = decoded;
                colIdx++;
            }
            rowIdx++;
        }

        // Blend horizontally within each row
        for (int r = 0; r < numRows; r++)
        {
            for (int c = 1; c < numCols; c++)
            {
                BlendHorizontal(tileGrid[r][c - 1], tileGrid[r][c], pixelBlendExtent);
            }
        }

        // Crop and concatenate horizontally
        Tensor[] rowTensors = new Tensor[numRows];
        for (int r = 0; r < numRows; r++)
        {
            rowTensors[r] = ConcatHorizontal(backend, tileGrid[r], pixelRowLimit, batch);

            // Dispose individual tiles
            for (int c = 0; c < numCols; c++)
            {
                tileGrid[r][c].Dispose();
            }
        }

        // Blend vertically between rows
        for (int r = 1; r < numRows; r++)
        {
            BlendVertical(rowTensors[r - 1], rowTensors[r], pixelBlendExtent);
        }

        // Crop and concatenate vertically
        Tensor result = ConcatVertical(backend, rowTensors, pixelRowLimit, batch);

        for (int r = 0; r < numRows; r++)
        {
            rowTensors[r].Dispose();
        }

        return result;
    }

    /// <summary>Extracts a tile [B, C, tileH, tileW] from latent at position (startH, startW).</summary>
    private static unsafe Tensor ExtractTile(Tensor source, int batch, int channels, int startH, int startW, int tileH, int tileW)
    {
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
                    int srcOffset = ((b * channels + c) * (int)source.Shape[2] + (startH + h)) * srcW + startW;
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

    /// <summary>Pads a tile to targetH x targetW with zeros.</summary>
    private static Tensor PadTile(IBackend backend, Tensor tile, int batch, int channels, int targetH, int targetW)
    {
        TensorShape paddedShape = new TensorShape(batch, channels, targetH, targetW);
        Tensor padded = new Tensor(paddedShape, DType.F32);
        backend.Fill(padded, 0f);

        unsafe
        {
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
        }

        return padded;
    }

    /// <summary>Crops a tile to cropH x cropW.</summary>
    private static unsafe Tensor CropTile(Tensor tile, int batch, int channels, int cropH, int cropW)
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

    /// <summary>Blends the overlap region between left and right tiles horizontally. Modifies right tile in-place.</summary>
    private static unsafe void BlendHorizontal(Tensor left, Tensor right, int blendExtent)
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

    /// <summary>Blends the overlap region between top and bottom tiles vertically. Modifies bottom tile in-place.</summary>
    private static unsafe void BlendVertical(Tensor top, Tensor bottom, int blendExtent)
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

    /// <summary>Concatenates tiles horizontally, cropping all but the last to rowLimit width.</summary>
    private static unsafe Tensor ConcatHorizontal(IBackend backend, Tensor[] tiles, int rowLimit, int batch)
    {
        int channels = (int)tiles[0].Shape[1];
        int height = (int)tiles[0].Shape[2];

        // Compute total width
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

    /// <summary>Concatenates rows vertically, cropping all but the last to rowLimit height.</summary>
    private static unsafe Tensor ConcatVertical(IBackend backend, Tensor[] rows, int rowLimit, int batch)
    {
        int channels = (int)rows[0].Shape[1];
        int width = (int)rows[0].Shape[3];

        // Compute total height
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
