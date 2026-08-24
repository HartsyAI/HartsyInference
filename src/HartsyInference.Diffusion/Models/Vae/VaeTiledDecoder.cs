using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Tiled VAE decoder that splits latents into overlapping tiles, decodes each independently, and blends with linear interpolation. Keeps VRAM usage constant regardless of output image size. Tile geometry and blending live in <see cref="VaeTiling"/>, shared with <see cref="VaeTiledEncoder"/>. Creates a tiled decoder wrapping the given VaeDecoder.</summary>
/// <param name="decoder">The underlying VAE decoder to use for individual tiles.</param>
/// <param name="tileOverlapFactor">Fraction of tile that overlaps with neighbors. Default: 0.25.</param>
public sealed class VaeTiledDecoder(VaeDecoder decoder, float tileOverlapFactor = 0.25f)
{
    private readonly VaeDecoder _decoder = decoder;
    private readonly float _tileOverlapFactor = tileOverlapFactor;
    private const int SpatialCompressionFactor = 8;

    /// <summary>Decodes a latent tensor using tiled decoding. Input: [B, C, H, W] (latent space). Output: [B, 3, H*8, W*8] (pixel space).</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
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

                Tensor tile = VaeTiling.ExtractTile(latent, batch, latentCh, i, j, tileH, tileW);

                // Pad tile to full tile size if needed (at edges)
                Tensor paddedTile;
                if (tileH < tileLatentSize || tileW < tileLatentSize)
                {
                    paddedTile = VaeTiling.PadTile(backend, tile, batch, latentCh, tileLatentSize, tileLatentSize);
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
                    Tensor cropped = VaeTiling.CropTile(decoded, batch, 3, decodedH, decodedW);
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
                VaeTiling.BlendHorizontal(tileGrid[r][c - 1], tileGrid[r][c], pixelBlendExtent);
            }
        }

        // Crop and concatenate horizontally
        Tensor[] rowTensors = new Tensor[numRows];
        for (int r = 0; r < numRows; r++)
        {
            rowTensors[r] = VaeTiling.ConcatHorizontal(tileGrid[r], pixelRowLimit, batch);

            // Dispose individual tiles
            for (int c = 0; c < numCols; c++)
            {
                tileGrid[r][c].Dispose();
            }
        }

        // Blend vertically between rows
        for (int r = 1; r < numRows; r++)
        {
            VaeTiling.BlendVertical(rowTensors[r - 1], rowTensors[r], pixelBlendExtent);
        }

        // Crop and concatenate vertically
        Tensor result = VaeTiling.ConcatVertical(rowTensors, pixelRowLimit, batch);

        for (int r = 0; r < numRows; r++)
        {
            rowTensors[r].Dispose();
        }

        return result;
    }
}
