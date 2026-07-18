using System.Text;

namespace HartsyInference.Cli.Infra;

/// <summary>Renders RGB pixel data inline in the terminal using Unicode upper-half-block glyphs (▀) with 24-bit
/// truecolor: each character cell packs two vertical pixels (foreground = top, background = bottom), so a cell grid
/// of W×H shows a W×2H pixel image. Pure C#, no image library and no terminal-specific graphics protocol.</summary>
public static class TerminalImage
{
    private const char UpperHalfBlock = '▀';

    /// <summary>Whether inline previews should be emitted: a real (non-redirected) stdout, truecolor not opted out via
    /// <c>NO_COLOR</c>, and not disabled via <c>HARTSY_NO_IMAGE=1</c>.</summary>
    public static bool IsSupported =>
        !Console.IsOutputRedirected
        && Environment.GetEnvironmentVariable("NO_COLOR") is null
        && Environment.GetEnvironmentVariable("HARTSY_NO_IMAGE") != "1";

    private const char LowerHalfBlock = '▄';

    /// <summary>Prints <paramref name="rgb"/> (row-major, top-to-bottom, 3 bytes/pixel R,G,B) scaled to fit
    /// <paramref name="maxCellWidth"/> columns while preserving aspect ratio. Indented by <paramref name="indent"/>
    /// spaces. When <paramref name="transparentKey"/> is set, pixels matching that exact color render as the terminal's
    /// own background (so a logo drops onto any theme cleanly). No-op when <see cref="IsSupported"/> is false.</summary>
    public static void Render(byte[] rgb, int width, int height, int maxCellWidth = 56, int indent = 2,
        (byte R, byte G, byte B)? transparentKey = null)
    {
        if (!IsSupported || width <= 0 || height <= 0 || rgb.Length < (long)width * height * 3)
            return;

        int budget = Math.Min(maxCellWidth, Math.Max(8, TerminalWidth() - indent - 1));
        int cols = Math.Min(width, budget);
        // A cell is ~twice as tall as wide and holds 2 pixels vertically, so rows ≈ cols * (h/w) / 2 keeps aspect.
        int rows = Math.Max(1, (int)Math.Round(cols * (height / (double)width) / 2.0));

        StringBuilder sb = new StringBuilder(cols * rows * 24);
        string pad = new string(' ', indent);
        for (int cy = 0; cy < rows; cy++)
        {
            sb.Append(pad);
            for (int cx = 0; cx < cols; cx++)
            {
                (byte tr, byte tg, byte tb) = Sample(rgb, width, height, cx, cy * 2, cols, rows * 2);
                (byte br, byte bg, byte bb) = Sample(rgb, width, height, cx, cy * 2 + 1, cols, rows * 2);
                bool topClear = transparentKey is { } k1 && tr == k1.R && tg == k1.G && tb == k1.B;
                bool botClear = transparentKey is { } k2 && br == k2.R && bg == k2.G && bb == k2.B;

                if (topClear && botClear)
                {
                    sb.Append("\x1b[0m ");
                }
                else if (botClear)
                {
                    // Only the top pixel is opaque: upper half block, default background shows through below.
                    sb.Append("\x1b[49m\x1b[38;2;").Append(tr).Append(';').Append(tg).Append(';').Append(tb).Append('m').Append(UpperHalfBlock);
                }
                else if (topClear)
                {
                    // Only the bottom pixel is opaque: lower half block over the default background.
                    sb.Append("\x1b[49m\x1b[38;2;").Append(br).Append(';').Append(bg).Append(';').Append(bb).Append('m').Append(LowerHalfBlock);
                }
                else
                {
                    sb.Append("\x1b[38;2;").Append(tr).Append(';').Append(tg).Append(';').Append(tb).Append('m');
                    sb.Append("\x1b[48;2;").Append(br).Append(';').Append(bg).Append(';').Append(bb).Append('m');
                    sb.Append(UpperHalfBlock);
                }
            }
            sb.Append("\x1b[0m\n");
        }
        Console.Out.Write(sb.ToString());
    }

    // Nearest-neighbor sample of source pixel mapped from the (dstX,dstY) cell in a dstW×dstH grid.
    private static (byte r, byte g, byte b) Sample(byte[] rgb, int srcW, int srcH, int dstX, int dstY, int dstW, int dstH)
    {
        int sx = Math.Min(srcW - 1, (int)((dstX + 0.5) * srcW / dstW));
        int sy = Math.Min(srcH - 1, (int)((dstY + 0.5) * srcH / dstH));
        int i = (sy * srcW + sx) * 3;
        return (rgb[i], rgb[i + 1], rgb[i + 2]);
    }

    private static int TerminalWidth()
    {
        try
        {
            int w = Console.WindowWidth;
            return w > 0 ? w : 80;
        }
        catch
        {
            return 80;
        }
    }
}
