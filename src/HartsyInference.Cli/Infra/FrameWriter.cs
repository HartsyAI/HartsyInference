namespace HartsyInference.Cli.Infra;

/// <summary>Writes a sequence of RGB24 video frames as numbered PNG files into a fresh subdirectory.</summary>
public static class FrameWriter
{
    /// <summary>Writes <paramref name="frames"/> (each <c>width*height*3</c> RGB bytes) into a new subdirectory of
    /// <paramref name="baseDir"/> named after <paramref name="slug"/>. Returns the created directory path.</summary>
    public static string WriteFrames(byte[][] frames, int width, int height, string baseDir, string slug)
    {
        Directory.CreateDirectory(baseDir);
        string dir = NextDir(baseDir, Slug.Make(slug));
        Directory.CreateDirectory(dir);
        for (int i = 0; i < frames.Length; i++)
            File.WriteAllBytes(Path.Combine(dir, $"frame_{i + 1:D4}.png"), PngEncoder.Encode(frames[i], width, height));
        return dir;
    }

    private static string NextDir(string baseDir, string slug)
    {
        for (int i = 1; i < 100000; i++)
        {
            string candidate = Path.Combine(baseDir, $"{slug}-frames-{i:D3}");
            if (!Directory.Exists(candidate))
                return candidate;
        }
        return Path.Combine(baseDir, $"{slug}-frames-{Guid.NewGuid():N}");
    }
}
