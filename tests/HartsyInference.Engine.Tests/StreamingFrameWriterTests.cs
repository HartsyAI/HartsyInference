using System.Runtime.CompilerServices;

namespace HartsyInference.Engine.Tests;

/// <summary><see cref="StreamingFrameWriter"/>: a stream's frames go to numbered PNGs the moment they are added, byte for byte what the
/// batch <see cref="FrameWriter"/> writes, and the writer keeps none of the pixels except the first frame's. That last point is the
/// reason it exists: an unbounded stream would otherwise hold every frame it ever decoded (2.7 MB each at 720p).</summary>
public sealed class StreamingFrameWriterTests : IDisposable
{
    private const int Width = 8, Height = 6;
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"streaming-frames-{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void Add_WritesEachFrameAtOnce_NumberedFromOne()
    {
        StreamingFrameWriter writer = new(_root, "A world!");
        Assert.Null(writer.OutputDirectory);
        Assert.False(Directory.Exists(_root), "The directory is created by the first frame, not by the writer.");

        writer.Add(Frame(0), Width, Height);
        Assert.NotNull(writer.OutputDirectory);
        Assert.StartsWith(_root, writer.OutputDirectory);
        Assert.EndsWith("-frames-001", writer.OutputDirectory);
        Assert.True(File.Exists(Path.Combine(writer.OutputDirectory, "frame_0001.png")), "frame 1 is on disk before the stream ends");

        writer.Add(Frame(1), Width, Height);
        writer.Add(Frame(2), Width, Height);
        Assert.Equal(3, writer.Count);
        Assert.Equal((Width, Height), (writer.Width, writer.Height));
        Assert.Equal(["frame_0001.png", "frame_0002.png", "frame_0003.png"],
            Directory.GetFiles(writer.OutputDirectory).Select(file => Path.GetFileName(file)!).Order().ToArray());
    }

    [Fact]
    public void Add_WritesTheSameFilesAsTheBatchWriter()
    {
        byte[][] frames = [Frame(0), Frame(1), Frame(2), Frame(3)];
        string batch = FrameWriter.WriteFrames(frames, Width, Height, _root, "batch");
        StreamingFrameWriter writer = new(_root, "stream");
        foreach (byte[] frame in frames)
            writer.Add(frame, Width, Height);

        for (int i = 1; i <= frames.Length; i++)
        {
            string name = $"frame_{i:D4}.png";
            Assert.Equal(File.ReadAllBytes(Path.Combine(batch, name)), File.ReadAllBytes(Path.Combine(writer.OutputDirectory!, name)));
        }
    }

    [Fact]
    public void Add_KeepsNoFrameButTheFirst()
    {
        StreamingFrameWriter writer = new(_root, "spool");
        byte[] first = Frame(0);
        writer.Add(first, Width, Height);
        WeakReference later = AddLaterFrames(writer);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        Assert.False(later.IsAlive, "A later frame is still reachable from the writer.");
        Assert.Same(first, writer.FirstFrame);
        Assert.Equal(3, writer.Count);
    }

    [Fact]
    public void Add_RefusesPixelsThatDoNotFillTheFrame_AndAFrameOfAnotherSize()
    {
        StreamingFrameWriter writer = new(_root, "bad");
        Assert.Throws<ArgumentException>(() => writer.Add(new byte[Width * Height * 3 - 1], Width, Height));
        Assert.Equal(0, writer.Count);
        Assert.Null(writer.OutputDirectory);

        writer.Add(Frame(0), Width, Height);
        Assert.Throws<ArgumentException>(() => writer.Add(new byte[4 * 4 * 3], 4, 4));
        Assert.Equal(1, writer.Count);
    }

    [Fact]
    public void Complete_WithNoFrames_Throws() =>
        Assert.Throws<InvalidOperationException>(() => new StreamingFrameWriter(_root, "empty").Complete(null, 16));

    [Fact]
    public void Complete_ReturnsTheDirectory_WithOrWithoutAnMp4()
    {
        StreamingFrameWriter writer = new(_root, "done");
        for (int i = 0; i < 4; i++)
            writer.Add(Frame(i), Width, Height);

        // ffmpeg may be absent on the machine running the tests; a failed mux is a degraded result, never a failure.
        VideoOutputWriter.Written written = writer.Complete(null, 16);
        Assert.Equal(writer.OutputDirectory, written.Directory);
        Assert.Null(written.AudioPath);
        if (written.Mp4Path is not null)
            Assert.True(File.Exists(written.Mp4Path));
    }

    [Fact]
    public void TwoWriters_WithTheSameSlug_GetSeparateDirectories()
    {
        StreamingFrameWriter first = new(_root, "same"), second = new(_root, "same");
        first.Add(Frame(0), Width, Height);
        second.Add(Frame(0), Width, Height);
        Assert.NotEqual(first.OutputDirectory, second.OutputDirectory);
        Assert.EndsWith("-frames-002", second.OutputDirectory);
    }

    // The frames a caller drops after adding them: built and added in a frame the JIT cannot extend past the return.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference AddLaterFrames(StreamingFrameWriter writer)
    {
        byte[] second = Frame(1), third = Frame(2);
        writer.Add(second, Width, Height);
        writer.Add(third, Width, Height);
        return new WeakReference(third);
    }

    // Distinct, deterministic pixels per frame so a swapped or repeated frame shows up in the bytes.
    private static byte[] Frame(int index)
    {
        byte[] rgb = new byte[Width * Height * 3];
        for (int i = 0; i < rgb.Length; i++)
            rgb[i] = (byte)((i * 7 + index * 31) & 0xFF);
        return rgb;
    }
}
