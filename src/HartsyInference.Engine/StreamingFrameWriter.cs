using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine;

/// <summary>Writes a video generation's frames to disk as they arrive, so a long or unbounded stream holds none of its pixels
/// in memory: each frame is encoded into its own numbered PNG the moment it is added, and the writer keeps only a count, the
/// frame size and the first frame (the artifact's preview). <see cref="Complete"/> then writes the soundtrack and muxes
/// <c>video.mp4</c> from the frames on disk, as <see cref="VideoOutputWriter.Write"/> does for a finished batch.</summary>
/// <remarks>The directory is created when the first frame arrives, so a stream that produces nothing leaves nothing behind.
/// A stream that stops early still leaves every frame it produced on disk. Not thread-safe: one consumer adds the frames in
/// order.</remarks>
public sealed class StreamingFrameWriter
{
    private readonly string _baseDir;
    private readonly string _slug;
    private string? _outputDirectory;
    private byte[]? _firstFrame;

    /// <summary>Creates a writer that will put its frames in a new subdirectory of <paramref name="baseDir"/> named after
    /// <paramref name="slug"/>.</summary>
    public StreamingFrameWriter(string baseDir, string slug)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(baseDir);
        ArgumentNullException.ThrowIfNull(slug);
        _baseDir = baseDir;
        _slug = slug;
    }

    /// <summary>The directory the frames are written to, or null until the first frame arrives.</summary>
    public string? OutputDirectory => _outputDirectory;

    /// <summary>Frames written so far.</summary>
    public int Count { get; private set; }

    /// <summary>Frame width in pixels, or 0 before the first frame.</summary>
    public int Width { get; private set; }

    /// <summary>Frame height in pixels, or 0 before the first frame.</summary>
    public int Height { get; private set; }

    /// <summary>The first frame's RGB24 pixels, kept as the artifact's preview, or null before the first frame. It is the only
    /// frame the writer holds.</summary>
    public byte[]? FirstFrame => _firstFrame;

    /// <summary>Encodes <paramref name="rgb"/> (<c>width*height*3</c> bytes) as the next numbered PNG and writes it. The writer
    /// does not keep the array, except for the first frame.</summary>
    /// <exception cref="ArgumentException">The pixels do not fill <paramref name="width"/> by <paramref name="height"/>, or the
    /// size differs from the first frame's.</exception>
    public void Add(byte[] rgb, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(rgb);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        if (rgb.LongLength != (long)width * height * 3)
            throw new ArgumentException($"Frame {Count + 1} holds {rgb.LongLength} bytes; {width}x{height} RGB24 needs {(long)width * height * 3}.",
                nameof(rgb));
        if (Count > 0 && (width != Width || height != Height))
            throw new ArgumentException($"Frame {Count + 1} is {width}x{height}; the stream began at {Width}x{Height}.", nameof(width));

        _outputDirectory ??= FrameWriter.CreateDirectory(_baseDir, _slug);
        FrameWriter.WriteFrame(_outputDirectory, Count, rgb, width, height);
        if (Count == 0)
        {
            _firstFrame = rgb;
            Width = width;
            Height = height;
        }

        Count++;
    }

    /// <summary>Finishes the video: <c>audio.wav</c> when <paramref name="audio"/> carries a soundtrack, then <c>video.mp4</c>
    /// muxed from the frames on disk (see <see cref="VideoOutputWriter.Finish"/>).</summary>
    /// <exception cref="InvalidOperationException">No frame was added.</exception>
    public VideoOutputWriter.Written Complete(AudioBuffer? audio, int fps)
    {
        if (_outputDirectory is null)
            throw new InvalidOperationException("No frames were written.");
        return VideoOutputWriter.Finish(_outputDirectory, audio, fps);
    }
}
