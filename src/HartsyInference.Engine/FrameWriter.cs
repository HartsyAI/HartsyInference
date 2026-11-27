using HartsyInference.Core.Diagnostics;

namespace HartsyInference.Engine;

/// <summary>Writes a sequence of RGB24 video frames as numbered PNG files into a fresh subdirectory.</summary>
public static class FrameWriter
{
    /// <summary>Writes <paramref name="frames"/> (each <c>width*height*3</c> RGB bytes) into a new subdirectory of
    /// <paramref name="baseDir"/> named after <paramref name="slug"/>. Returns the created directory path.</summary>
    public static string WriteFrames(byte[][] frames, int width, int height, string baseDir, string slug)
    {
        string dir = CreateDirectory(baseDir, slug);
        for (int i = 0; i < frames.Length; i++)
            WriteFrame(dir, i, frames[i], width, height);
        return dir;
    }

    /// <summary>Creates a new, empty subdirectory of <paramref name="baseDir"/> named after <paramref name="slug"/> and
    /// returns its path. A stream that writes its frames as they arrive creates its directory once, then calls
    /// <see cref="WriteFrame"/> for each frame.</summary>
    public static string CreateDirectory(string baseDir, string slug)
    {
        Directory.CreateDirectory(baseDir);
        string dir = NextDir(baseDir, Slug.Make(slug));
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>Writes the frame at zero-based position <paramref name="index"/> (<c>width*height*3</c> RGB bytes) as
    /// <c>frame_NNNN.png</c>, numbered from 1: the pattern the MP4 mux reads.</summary>
    public static void WriteFrame(string dir, int index, byte[] rgb, int width, int height)
    {
        long encodeStart = WorldStageTimings.Start();
        byte[] png = PngEncoder.Encode(rgb, width, height);
        WorldStageTimings.Record(WorldStage.FrameEncode, encodeStart);
        long writeStart = WorldStageTimings.Start();
        File.WriteAllBytes(Path.Combine(dir, $"frame_{index + 1:D4}.png"), png);
        WorldStageTimings.Record(WorldStage.FrameWrite, writeStart);
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
