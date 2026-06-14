namespace HartsyInference.Video.Encoding;

/// <summary>Consumes a stream of <see cref="VideoFrame"/>s and writes a video artifact (MP4, frame sequence, …). Pull-based: the encoder drives the async enumerable, so decode is naturally paced by encoding (backpressure).</summary>
public interface IVideoEncoder
{
    /// <summary>Encodes the frame stream to <paramref name="outputPath"/> at <paramref name="fps"/> frames/second.</summary>
    Task EncodeAsync(IAsyncEnumerable<VideoFrame> frames, string outputPath, int fps, CancellationToken cancellationToken = default);
}
