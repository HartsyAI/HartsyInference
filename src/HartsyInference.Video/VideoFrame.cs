namespace HartsyInference.Video;

/// <summary>A single decoded video frame: interleaved RGB bytes [0,255] plus its dimensions and sequence index. Streamed by the video pipelines and consumed by an <see cref="Encoding.IVideoEncoder"/>.</summary>
public readonly record struct VideoFrame(int Index, int Width, int Height, byte[] Rgb);
