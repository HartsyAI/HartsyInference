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
/// <param name="backend">Compute backend.</param>
/// <param name="net">Loaded RRDBNet generator.</param>
/// <param name="inputTileSize">Tile size in input pixels (0 disables tiling — whole image at once). Default 128.</param>
/// <param name="tileOverlapFactor">Fraction of a tile that overlaps its neighbours. Default 0.25.</param>
public sealed class UpscalePipeline(IBackend backend, RrdbNet net, int inputTileSize = 128,
    float tileOverlapFactor = 0.25f) : IUpscalePipeline
{
    private readonly IBackend _backend = backend;
    private readonly RrdbNet _net = net;
    private readonly int _inputTileSize = inputTileSize;
    private readonly float _tileOverlapFactor = tileOverlapFactor;

    /// <inheritdoc/>
    public string ModelName => $"real-esrgan-x{_net.Config.Scale}";

    /// <inheritdoc/>
    public int ScaleFactor => _net.Config.Scale;

    /// <summary>The network's own spatial factor over whatever it is fed: always 4 (see <see cref="RrdbNet"/>).</summary>
    private const int NetScale = 4;

    /// <inheritdoc/>
    public (byte[] rgbData, int width, int height) Upscale(ReadOnlySpan<byte> rgbData, int width, int height)
    {
        int r = _net.Config.UnshuffleFactor;
        // A 2× model consumes the image pixel-unshuffled by 2, so odd edges are replicated out to even first and the
        // surplus is cropped off the output (Real-ESRGAN pads to a multiple of the factor the same way).
        int padW = (r - width % r) % r, padH = (r - height % r) % r;
        Tensor input = r == 1
            ? ImageTensor.RgbToTensor01(rgbData, width, height)
            : UnshuffledInput(rgbData, width, height, r, padW, padH);
        int inW = (width + padW) / r, inH = (height + padH) / r;
        Tensor output = _inputTileSize <= 0
            ? _net.Forward(_backend, input)
            : TiledForward(input, inW, inH, _net.Config.InputChannels, Math.Max(8, _inputTileSize / r));
        input.Dispose();

        int outW = width * ScaleFactor;
        int outH = height * ScaleFactor;
        byte[] bytes = Tensor01ToRgb(output, outW, outH);
        output.Dispose();
        return (bytes, outW, outH);
    }

    /// <summary>Overlapping-tile forward. Mirrors <see cref="VaeTiledDecoder"/>: tile in (possibly unshuffled) input
    /// space, upscale each tile by <see cref="NetScale"/>, blend in output space with a linear ramp.</summary>
    private Tensor TiledForward(Tensor input, int width, int height, int channels, int tile)
    {
        const int scale = NetScale;

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

                Tensor t = VaeTiling.ExtractTile(input, 1, channels, i, j, tileH, tileW);
                Tensor padded = tileH < tile || tileW < tile ? VaeTiling.PadTile(_backend, t, 1, channels, tile, tile) : t;
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

    /// <summary>RGB bytes → the 2× (or r×) pixel-unshuffled NCHW F32 tensor BasicSR's 2× RRDBNet expects:
    /// <c>out[c·r² + i·r + j, y, x] = in[c, y·r + i, x·r + j]</c> (torch <c>pixel_unshuffle</c> channel order), with
    /// the right/bottom edge replicated by <paramref name="padW"/> / <paramref name="padH"/> pixels so the source
    /// divides evenly.</summary>
    internal static unsafe Tensor UnshuffledInput(ReadOnlySpan<byte> rgb, int width, int height, int r, int padW, int padH)
    {
        int fullW = width + padW, fullH = height + padH;
        int outW = fullW / r, outH = fullH / r;
        int channels = 3 * r * r;
        Tensor t = new Tensor(new TensorShape(1, channels, outH, outW), DType.F32);
        float* d = (float*)t.DataPointer;
        long plane = (long)outW * outH;
        for (int y = 0; y < fullH; y++)
        {
            int sy = Math.Min(y, height - 1);
            for (int x = 0; x < fullW; x++)
            {
                int sx = Math.Min(x, width - 1);
                int src = (sy * width + sx) * 3;
                int oy = y / r, ox = x / r, i = y % r, j = x % r;
                long dst = (long)oy * outW + ox;
                for (int c = 0; c < 3; c++)
                {
                    int channel = c * r * r + i * r + j;
                    d[channel * plane + dst] = rgb[src + c] / 255f;
                }
            }
        }
        return t;
    }

    /// <summary>NCHW F32 [1,3,H,W] in [0,1] → HWC bytes [0,255] (clamped), reading only the top-left
    /// <paramref name="cropW"/> × <paramref name="cropH"/> window (the rest is edge padding from an odd-sized source).</summary>
    private static unsafe byte[] Tensor01ToRgb(Tensor t, int cropW, int cropH)
    {
        int height = (int)t.Shape[2];
        int width = (int)t.Shape[3];
        if (cropW > width || cropH > height)
            throw new ArgumentException($"Crop {cropW}x{cropH} exceeds the network output {width}x{height}.");
        int plane = width * height;
        byte[] rgb = new byte[cropW * cropH * 3];
        float* s = (float*)t.DataPointer;
        for (int y = 0; y < cropH; y++)
        {
            for (int x = 0; x < cropW; x++)
            {
                int src = y * width + x;
                int px = (y * cropW + x) * 3;
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
