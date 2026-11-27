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
        writer.WaitForWrites();
        Assert.True(File.Exists(Path.Combine(writer.OutputDirectory, "frame_0001.png")), "frame 1 is on disk once the writers have it");

        writer.Add(Frame(1), Width, Height);
        writer.Add(Frame(2), Width, Height);
        writer.WaitForWrites();
        Assert.Equal(3, writer.Count);
        Assert.Equal((Width, Height), (writer.Width, writer.Height));
        Assert.Equal(["frame_0001.png", "frame_0002.png", "frame_0003.png"],
            Directory.GetFiles(writer.OutputDirectory).Select(file => Path.GetFileName(file)!).Order().ToArray());
    }

    [Fact]
    public void Add_WritesTheSameFilesAsTheBatchWriter()
    {
        // More frames than the queue holds, so the writers run past the point where the stream has to wait for them.
        byte[][] frames = Enumerable.Range(0, StreamingFrameWriter.QueueCapacity + 8).Select(Frame).ToArray();
        string batch = FrameWriter.WriteFrames(frames, Width, Height, _root, "batch");
        StreamingFrameWriter writer = new(_root, "stream");
        foreach (byte[] frame in frames)
            writer.Add(frame, Width, Height);
        writer.WaitForWrites();

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
        writer.WaitForWrites();

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
    public async Task A_stalled_writer_holds_the_stream_back_once_the_queue_is_full()
    {
        using ManualResetEventSlim gate = new(initialState: false);
        StreamingFrameWriter writer = new(_root, "held", gate);
        const int Total = StreamingFrameWriter.QueueCapacity + 16;
        Task adding = Task.Run(() =>
        {
            for (int i = 0; i < Total; i++)
                writer.Add(Frame(i), Width, Height);
        });

        await Task.Delay(300);
        Assert.False(adding.IsCompleted, "the stream must wait while the writers are held");
        // The queue holds QueueCapacity frames, the writers hold a few more, and one is waiting to enter the queue: bounded, not growing.
        Assert.InRange(writer.Count, StreamingFrameWriter.QueueCapacity, StreamingFrameWriter.QueueCapacity + 16);

        gate.Set();
        await adding.WaitAsync(TimeSpan.FromSeconds(10));
        writer.WaitForWrites();
        Assert.Equal(Total, writer.Count);
        Assert.Equal(Total, Directory.GetFiles(writer.OutputDirectory!, "*.png").Length);
        writer.Dispose();
    }

    [Fact]
    public void Dispose_waits_for_the_queued_frames_so_a_stream_that_stops_early_keeps_them()
    {
        StreamingFrameWriter writer = new(_root, "stopped");
        for (int i = 0; i < 5; i++)
            writer.Add(Frame(i), Width, Height);

        writer.Dispose();

        Assert.Equal(5, Directory.GetFiles(writer.OutputDirectory!, "*.png").Length);
    }

    [Fact]
    public void A_frame_that_cannot_be_written_fails_the_next_wait_and_the_stream_after_it()
    {
        StreamingFrameWriter writer = new(_root, "gone");
        writer.Add(Frame(0), Width, Height);
        writer.WaitForWrites();
        Directory.Delete(writer.OutputDirectory!, recursive: true);

        writer.Add(Frame(1), Width, Height);

        Assert.Throws<InvalidOperationException>(() => writer.WaitForWrites());
        Assert.Throws<InvalidOperationException>(() => writer.Add(Frame(2), Width, Height));
        writer.Dispose();
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
