namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Decoded source video: interleaved RGB24 frames in presentation order with their timestamps in seconds
/// (PyAV <c>frame.time</c> in the official loader).</summary>
public sealed record ControlFoleyRawVideo
{
    /// <summary>One <c>Height * Width * 3</c> byte array per frame.</summary>
    public required IReadOnlyList<byte[]> Frames { get; init; }

    /// <summary>Presentation time of each frame in seconds.</summary>
    public required IReadOnlyList<double> Times { get; init; }

    /// <summary>Frame width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Frame height in pixels.</summary>
    public required int Height { get; init; }

    /// <summary>Frames at a constant rate starting at time zero.</summary>
    public static ControlFoleyRawVideo FromConstantRate(IReadOnlyList<byte[]> frames, int width, int height, double fps)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (fps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(fps), "The frame rate must be positive.");
        }

        double[] times = new double[frames.Count];
        for (int i = 0; i < times.Length; i++)
        {
            times[i] = i / fps;
        }

        return new ControlFoleyRawVideo { Frames = frames, Times = times, Width = width, Height = height };
    }
}
