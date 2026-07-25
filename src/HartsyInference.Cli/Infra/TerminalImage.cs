using System.Text;
using HartsyInference.Core.Logging;

namespace HartsyInference.Cli.Infra;

/// <summary>Renders RGB pixel data inline in the terminal using Unicode half-block glyphs with 24-bit truecolor.</summary>
/// <remarks>Each character cell packs two vertical pixels (foreground = top, background = bottom), so a cell grid of W×H
/// shows a W×2H pixel image. Downsampling is area-averaged (box filter) with an optional unsharp pass, which is far
/// cleaner than nearest-neighbor when shrinking a photo or logo. Pure C#, no image library and no graphics protocol.</remarks>
public static class TerminalImage
{
    private const char UpperHalfBlock = '▀';
    private const char LowerHalfBlock = '▄';

    /// <summary>Whether inline previews should be emitted.</summary>
    /// <remarks>Requires a real (non-redirected) stdout, truecolor not opted out via <c>NO_COLOR</c>, and not disabled via
    /// <c>HARTSY_NO_IMAGE=1</c>.</remarks>
    public static bool IsSupported =>
        !Console.IsOutputRedirected
        && Environment.GetEnvironmentVariable("NO_COLOR") is null
        && Environment.GetEnvironmentVariable("HARTSY_NO_IMAGE") != "1";

    /// <summary>Prints <paramref name="rgb"/> scaled to fit <paramref name="maxCellWidth"/> columns; no-op when unsupported.</summary>
    /// <remarks>See <see cref="RenderToLines"/> for the parameter meanings.</remarks>
    public static void Render(byte[] rgb, int width, int height, int maxCellWidth = 56, int indent = 2,
        (byte R, byte G, byte B)? transparentKey = null, float sharpen = 0f)
    {
        List<string>? lines = RenderToLines(rgb, width, height, maxCellWidth, indent, transparentKey, sharpen);
        if (lines is null)
            return;
        Console.Out.Write(string.Join('\n', lines));
        Console.Out.Write('\n');
    }

    /// <summary>Renders <paramref name="rgb"/> (RGB24) to one ANSI string per row, scaled to <paramref name="maxCellWidth"/> columns.</summary>
    /// <remarks>Each row is already indented and color-reset with no trailing newline, so a caller can lay text out beside
    /// it. When <paramref name="transparentKey"/> is set, source pixels matching that exact color render as the terminal's
    /// own background. <paramref name="sharpen"/> (0..1) applies a light unsharp pass to crisp edges. Null when
    /// <see cref="IsSupported"/> is false or the input is invalid.</remarks>
    public static List<string>? RenderToLines(byte[] rgb, int width, int height, int maxCellWidth = 56, int indent = 2,
        (byte R, byte G, byte B)? transparentKey = null, float sharpen = 0f)
    {
        if (!IsSupported || width <= 0 || height <= 0 || rgb.Length < (long)width * height * 3)
            return null;

        int budget = Math.Min(maxCellWidth, Math.Max(8, TerminalWidth() - indent - 1));
        int cols = Math.Min(width, budget);
        // A cell is ~twice as tall as wide and holds 2 pixels vertically, so rows ≈ cols * (h/w) / 2 keeps aspect.
        int rows = Math.Max(1, (int)Math.Round(cols * (height / (double)width) / 2.0));
        int gridH = rows * 2;

        Pixel[] grid = Downsample(rgb, width, height, cols, gridH, transparentKey);
        if (sharpen > 0f)
            grid = Sharpen(grid, cols, gridH, sharpen);

        List<string> lines = new List<string>(rows);
        string pad = new string(' ', indent);
        for (int cy = 0; cy < rows; cy++)
        {
            StringBuilder sb = new StringBuilder(cols * 24);
            sb.Append(pad);
            for (int cx = 0; cx < cols; cx++)
            {
                Pixel top = grid[(cy * 2) * cols + cx];
                Pixel bot = grid[(cy * 2 + 1) * cols + cx];
                if (top.Clear && bot.Clear)
                    sb.Append("\x1b[0m ");
                else if (bot.Clear)
                    sb.Append("\x1b[49m\x1b[38;2;").Append(top.R).Append(';').Append(top.G).Append(';').Append(top.B).Append('m').Append(UpperHalfBlock);
                else if (top.Clear)
                    sb.Append("\x1b[49m\x1b[38;2;").Append(bot.R).Append(';').Append(bot.G).Append(';').Append(bot.B).Append('m').Append(LowerHalfBlock);
                else
                {
                    sb.Append("\x1b[38;2;").Append(top.R).Append(';').Append(top.G).Append(';').Append(top.B).Append('m');
                    sb.Append("\x1b[48;2;").Append(bot.R).Append(';').Append(bot.G).Append(';').Append(bot.B).Append('m');
                    sb.Append(UpperHalfBlock);
                }
            }
            sb.Append("\x1b[0m");
            lines.Add(sb.ToString());
        }
        return lines;
    }

    private readonly record struct Pixel(byte R, byte G, byte B, bool Clear);

    // Box-filter downsample: each target pixel averages the RGB of the source region mapping to it. With a transparent
    // key, only non-key source pixels contribute and a target is "clear" when they are the minority — coverage-weighted
    // (premultiplied) downsampling, so logo edges anti-alias instead of aliasing.
    private static Pixel[] Downsample(byte[] rgb, int srcW, int srcH, int dstW, int dstH, (byte R, byte G, byte B)? key)
    {
        Pixel[] grid = new Pixel[dstW * dstH];
        for (int dy = 0; dy < dstH; dy++)
        {
            int sy0 = (int)((long)dy * srcH / dstH);
            int sy1 = Math.Max(sy0 + 1, (int)((long)(dy + 1) * srcH / dstH));
            for (int dx = 0; dx < dstW; dx++)
            {
                int sx0 = (int)((long)dx * srcW / dstW);
                int sx1 = Math.Max(sx0 + 1, (int)((long)(dx + 1) * srcW / dstW));
                long sr = 0, sg = 0, sb = 0;
                int opaque = 0, total = 0;
                for (int sy = sy0; sy < sy1 && sy < srcH; sy++)
                {
                    int rowBase = sy * srcW;
                    for (int sx = sx0; sx < sx1 && sx < srcW; sx++)
                    {
                        int i = (rowBase + sx) * 3;
                        byte r = rgb[i], g = rgb[i + 1], b = rgb[i + 2];
                        total++;
                        if (key is { } k && r == k.R && g == k.G && b == k.B)
                            continue;
                        sr += r; sg += g; sb += b; opaque++;
                    }
                }
                if (key is not null && opaque * 2 < total)
                    grid[dy * dstW + dx] = new Pixel(0, 0, 0, true);
                else if (opaque > 0)
                    grid[dy * dstW + dx] = new Pixel((byte)(sr / opaque), (byte)(sg / opaque), (byte)(sb / opaque), false);
                else
                    grid[dy * dstW + dx] = new Pixel(0, 0, 0, key is not null);
            }
        }
        return grid;
    }

    // Unsharp mask: push each opaque pixel away from the mean of its opaque 4-neighbors to crisp strokes and edges.
    // Clear (transparent) pixels and their contributions are skipped so the keyed background stays clean.
    private static Pixel[] Sharpen(Pixel[] grid, int w, int h, float amount)
    {
        Pixel[] outGrid = (Pixel[])grid.Clone();
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                Pixel p = grid[y * w + x];
                if (p.Clear)
                    continue;
                int nr = 0, ng = 0, nb = 0, n = 0;
                Accumulate(grid, w, h, x - 1, y, ref nr, ref ng, ref nb, ref n);
                Accumulate(grid, w, h, x + 1, y, ref nr, ref ng, ref nb, ref n);
                Accumulate(grid, w, h, x, y - 1, ref nr, ref ng, ref nb, ref n);
                Accumulate(grid, w, h, x, y + 1, ref nr, ref ng, ref nb, ref n);
                if (n == 0)
                    continue;
                float mr = nr / (float)n, mg = ng / (float)n, mb = nb / (float)n;
                outGrid[y * w + x] = new Pixel(
                    ClampByte(p.R + amount * (p.R - mr)),
                    ClampByte(p.G + amount * (p.G - mg)),
                    ClampByte(p.B + amount * (p.B - mb)), false);
            }
        }
        return outGrid;
    }

    private static void Accumulate(Pixel[] grid, int w, int h, int x, int y, ref int r, ref int g, ref int b, ref int n)
    {
        if (x < 0 || y < 0 || x >= w || y >= h)
            return;
        Pixel p = grid[y * w + x];
        if (p.Clear)
            return;
        r += p.R; g += p.G; b += p.B; n++;
    }

    private static byte ClampByte(float v) => (byte)Math.Clamp((int)MathF.Round(v), 0, 255);

    private static int TerminalWidth()
    {
        try
        {
            int w = Console.WindowWidth;
            return w > 0 ? w : 80;
        }
        catch (Exception ex)
        {
            Logs.Warning($"Could not read terminal width: {ex.Message}");
            return 80;
        }
    }
}
