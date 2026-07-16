using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Vision.Upscale;

/// <summary>End-to-end Real-ESRGAN / ESRGAN super-resolution. Wraps an <see cref="RrdbNet"/> with RGB
/// pre/post-processing and overlapping-tile inference so arbitrary-size images upscale at bounded VRAM.
/// Tiling reuses the same tested linear-ramp blender as the VAE (<see cref="VaeTiling"/>); here the
/// spatial factor is the model's upscale ratio rather than the VAE's 8x.</summary>
public sealed class UpscalePipeline : IUpscalePipeline
{
    private readonly IBackend _backend;
    private readonly RrdbNet _net;
    private readonly int _inputTileSize;
    private readonly float _tileOverlapFactor;

    /// <inheritdoc/>
    public string ModelName => $"real-esrgan-x{_net.Config.Scale}";

    /// <inheritdoc/>
    public int ScaleFactor => _net.Config.Scale;

    /// <summary>Creates an upscale pipeline.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="net">Loaded RRDBNet generator.</param>
    /// <param name="inputTileSize">Tile size in input pixels (0 disables tiling — whole image at once). Default 128.</param>
    /// <param name="tileOverlapFactor">Fraction of a tile that overlaps its neighbours. Default 0.25.</param>
    public UpscalePipeline(IBackend backend, RrdbNet net, int inputTileSize = 128, float tileOverlapFactor = 0.25f)
    {
        _backend = backend;
        _net = net;
        _inputTileSize = inputTileSize;
        _tileOverlapFactor = tileOverlapFactor;
    }

    /// <inheritdoc/>
    public (byte[] rgbData, int width, int height) Upscale(ReadOnlySpan<byte> rgbData, int width, int height)
    {
        Tensor input = ImageTensor.RgbToTensor01(rgbData, width, height);
        Tensor output = _inputTileSize <= 0
            ? _net.Forward(_backend, input)
            : TiledForward(input, width, height);
        input.Dispose();

        byte[] bytes = Tensor01ToRgb(output);
        int outW = width * ScaleFactor;
        int outH = height * ScaleFactor;
        output.Dispose();
        return (bytes, outW, outH);
    }

    /// <summary>Overlapping-tile forward. Mirrors <see cref="VaeTiledDecoder"/>: tile in input space,
    /// upscale each tile, blend in output space with a linear ramp.</summary>
    private Tensor TiledForward(Tensor input, int width, int height)
    {
        int scale = ScaleFactor;
        int tile = _inputTileSize;

        if (width <= tile && height <= tile)
        {
            return _net.Forward(_backend, input);
        }

        int overlapStep = (int)(tile * (1.0f - _tileOverlapFactor));
        int blendExtent = (int)(tile * _tileOverlapFactor);
        int rowLimit = tile - blendExtent;

        int outBlendExtent = blendExtent * scale;
        int outRowLimit = rowLimit * scale;
        int outTileSize = tile * scale;

        int numRows = 0;
        for (int i = 0; i < height; i += overlapStep) numRows++;
        int numCols = 0;
        for (int j = 0; j < width; j += overlapStep) numCols++;

        Tensor[][] grid = new Tensor[numRows][];
        int rowIdx = 0;
        for (int i = 0; i < height; i += overlapStep)
        {
            grid[rowIdx] = new Tensor[numCols];
            int colIdx = 0;
            for (int j = 0; j < width; j += overlapStep)
            {
                int tileH = Math.Min(tile, height - i);
                int tileW = Math.Min(tile, width - j);

                Tensor t = VaeTiling.ExtractTile(input, 1, 3, i, j, tileH, tileW);
                Tensor padded = tileH < tile || tileW < tile
                    ? VaeTiling.PadTile(_backend, t, 1, 3, tile, tile)
                    : t;
                if (!ReferenceEquals(padded, t)) t.Dispose();

                Tensor up = _net.Forward(_backend, padded);
                padded.Dispose();

                int upH = tileH * scale;
                int upW = tileW * scale;
                if (upH < outTileSize || upW < outTileSize)
                {
                    Tensor cropped = VaeTiling.CropTile(up, 1, 3, upH, upW);
                    up.Dispose();
                    up = cropped;
                }

                grid[rowIdx][colIdx] = up;
                colIdx++;
            }
            rowIdx++;
        }

        for (int r = 0; r < numRows; r++)
        {
            for (int c = 1; c < numCols; c++)
            {
                VaeTiling.BlendHorizontal(grid[r][c - 1], grid[r][c], outBlendExtent);
            }
        }

        Tensor[] rows = new Tensor[numRows];
        for (int r = 0; r < numRows; r++)
        {
            rows[r] = VaeTiling.ConcatHorizontal(grid[r], outRowLimit, 1);
            for (int c = 0; c < numCols; c++) grid[r][c].Dispose();
        }

        for (int r = 1; r < numRows; r++)
        {
            VaeTiling.BlendVertical(rows[r - 1], rows[r], outBlendExtent);
        }

        Tensor result = VaeTiling.ConcatVertical(rows, outRowLimit, 1);
        for (int r = 0; r < numRows; r++) rows[r].Dispose();
        return result;
    }

    /// <summary>NCHW F32 [1,3,H,W] in [0,1] → HWC bytes [0,255] (clamped).</summary>
    private static unsafe byte[] Tensor01ToRgb(Tensor t)
    {
        int height = (int)t.Shape[2];
        int width = (int)t.Shape[3];
        int plane = width * height;
        byte[] rgb = new byte[plane * 3];
        float* s = (float*)t.DataPointer;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int src = y * width + x;
                int px = (y * width + x) * 3;
                rgb[px] = ToByte(s[src]);
                rgb[px + 1] = ToByte(s[plane + src]);
                rgb[px + 2] = ToByte(s[2 * plane + src]);
            }
        }
        return rgb;
    }

    private static byte ToByte(float v) => (byte)Math.Clamp((int)MathF.Round(v * 255f), 0, 255);

    /// <inheritdoc/>
    public void Dispose() { }
}
