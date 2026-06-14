using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Encoding;

/// <summary>Dependency-free <see cref="IVideoEncoder"/> that writes each frame as <c>frame_NNNNN.bmp</c> under <paramref name="outputPath"/> (treated as a directory). Reuses <see cref="ImagePostProcessor.SaveBmp"/>. Useful when ffmpeg isn't available or for inspecting raw frames; <c>fps</c> is ignored.</summary>
public sealed class BmpSequenceEncoder : IVideoEncoder
{
    /// <inheritdoc/>
    public async Task EncodeAsync(IAsyncEnumerable<VideoFrame> frames, string outputPath, int fps, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(outputPath);
        await foreach (VideoFrame f in frames.WithCancellation(cancellationToken))
            ImagePostProcessor.SaveBmp(Path.Combine(outputPath, $"frame_{f.Index:D5}.bmp"), f.Rgb, f.Width, f.Height);
    }
}
