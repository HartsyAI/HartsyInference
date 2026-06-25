using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Tiled VAE encoder that splits an image into overlapping tiles, encodes each independently,
/// and blends the resulting latents with linear interpolation. Keeps VRAM usage constant regardless of
/// input resolution, unblocking high-res img2img / hires-fix where a single-shot encode's im2col
/// workspace would blow up.
///
/// <para>This is the inverse of <see cref="VaeTiledDecoder"/>: the decoder tiles in latent space and
/// blends in pixel space; the encoder tiles in <b>pixel</b> space and blends in the smaller
/// <b>latent</b> space (8x cheaper memory for the blend). Tile geometry and blending live in the shared
/// <see cref="VaeTiling"/> helper.</para></summary>
public sealed class VaeTiledEncoder
{
    private readonly VaeEncoder _encoder;
    private readonly float _tileOverlapFactor;
    private const int SpatialCompressionFactor = 8;

    /// <summary>Creates a tiled encoder wrapping the given VaeEncoder.</summary>
    /// <param name="encoder">The underlying VAE encoder used for individual tiles.</param>
    /// <param name="tileOverlapFactor">Fraction of tile that overlaps with neighbors. Default: 0.25 (matches <see cref="VaeTiledDecoder"/>).</param>
    public VaeTiledEncoder(VaeEncoder encoder, float tileOverlapFactor = 0.25f)
    {
        _encoder = encoder;
        _tileOverlapFactor = tileOverlapFactor;
    }

    /// <summary>Encodes an image using tiled encoding. Input: [B, 3, H, W] (pixel space, values in [-1, 1]).
    /// Output: [B, latentCh, H/8, W/8] (scaled latent space). Bit-identical to a single-shot
    /// <see cref="VaeEncoder.Encode"/> when the image fits a single tile.</summary>
    public Tensor Encode(IBackend backend, Tensor image)
    {
        VaeConfig config = _encoder.Config;
        int batch = (int)image.Shape[0];
        int imageCh = (int)image.Shape[1]; // 3
        int imageH = (int)image.Shape[2];
        int imageW = (int)image.Shape[3];

        // Tile size in latent space, then the corresponding pixel-space geometry.
        int tileLatentSize = config.SampleSize / (int)Math.Pow(2, config.BlockOutChannels.Length - 1);
        int latentOverlapStep = (int)(tileLatentSize * (1.0f - _tileOverlapFactor));
        int latentBlendExtent = (int)(tileLatentSize * _tileOverlapFactor);
        int latentRowLimit = tileLatentSize - latentBlendExtent;

        int pixelTileSize = tileLatentSize * SpatialCompressionFactor;
        int pixelOverlapStep = latentOverlapStep * SpatialCompressionFactor;

        // If the image fits in a single tile, encode directly (exact, no blending).
        if (imageH <= pixelTileSize && imageW <= pixelTileSize)
        {
            return _encoder.Encode(backend, image);
        }

        int latentCh = config.LatentChannels;

        // Walk the image in pixel space, encode each tile into latent space.
        int numRows = 0;
        for (int i = 0; i < imageH; i += pixelOverlapStep) numRows++;

        int numCols = 0;
        for (int j = 0; j < imageW; j += pixelOverlapStep) numCols++;

        Tensor[][] tileGrid = new Tensor[numRows][];

        int rowIdx = 0;
        for (int i = 0; i < imageH; i += pixelOverlapStep)
        {
            tileGrid[rowIdx] = new Tensor[numCols];
            int colIdx = 0;

            for (int j = 0; j < imageW; j += pixelOverlapStep)
            {
                // Extract pixel tile (clamp to bounds). H/W stay multiples of 8 because both the
                // image dims and pixelOverlapStep are multiples of the compression factor.
                int tilePixH = Math.Min(pixelTileSize, imageH - i);
                int tilePixW = Math.Min(pixelTileSize, imageW - j);

                Tensor tile = VaeTiling.ExtractTile(image, batch, imageCh, i, j, tilePixH, tilePixW);

                // Pad edge tiles to a full tile so the encoder always sees a fixed size.
                Tensor paddedTile;
                if (tilePixH < pixelTileSize || tilePixW < pixelTileSize)
                {
                    paddedTile = VaeTiling.PadTile(backend, tile, batch, imageCh, pixelTileSize, pixelTileSize);
                    tile.Dispose();
                }
                else
                {
                    paddedTile = tile;
                }

                // Encode this tile → latent [B, latentCh, tileLatentSize, tileLatentSize].
                Tensor encoded = _encoder.Encode(backend, paddedTile);
                paddedTile.Dispose();

                // Crop the latent back to the un-padded extent.
                int latentTileH = tilePixH / SpatialCompressionFactor;
                int latentTileW = tilePixW / SpatialCompressionFactor;
                if (latentTileH < tileLatentSize || latentTileW < tileLatentSize)
                {
                    Tensor cropped = VaeTiling.CropTile(encoded, batch, latentCh, latentTileH, latentTileW);
                    encoded.Dispose();
                    encoded = cropped;
                }

                tileGrid[rowIdx][colIdx] = encoded;
                colIdx++;
            }
            rowIdx++;
        }

        // Blend horizontally within each row (latent space).
        for (int r = 0; r < numRows; r++)
        {
            for (int c = 1; c < numCols; c++)
            {
                VaeTiling.BlendHorizontal(tileGrid[r][c - 1], tileGrid[r][c], latentBlendExtent);
            }
        }

        // Crop and concatenate horizontally.
        Tensor[] rowTensors = new Tensor[numRows];
        for (int r = 0; r < numRows; r++)
        {
            rowTensors[r] = VaeTiling.ConcatHorizontal(tileGrid[r], latentRowLimit, batch);
            for (int c = 0; c < numCols; c++)
            {
                tileGrid[r][c].Dispose();
            }
        }

        // Blend vertically between rows.
        for (int r = 1; r < numRows; r++)
        {
            VaeTiling.BlendVertical(rowTensors[r - 1], rowTensors[r], latentBlendExtent);
        }

        // Crop and concatenate vertically.
        Tensor result = VaeTiling.ConcatVertical(rowTensors, latentRowLimit, batch);

        for (int r = 0; r < numRows; r++)
        {
            rowTensors[r].Dispose();
        }

        return result;
    }
}
