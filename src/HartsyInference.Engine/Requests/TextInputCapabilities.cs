namespace HartsyInference.Engine.Requests;

/// <summary>What a local text model takes as input beyond text, read from the model file or folder alone (nothing is loaded). A host
/// uses it to enable or disable image and video attachments without carrying its own list of model families.</summary>
/// <param name="Images">Whether the model answers about images: a GGUF with a sidecar <c>mmproj</c>, or a MOSS-VL model.</param>
/// <param name="Video">Whether the model takes a video (<see cref="TextMessage.Video"/> or <see cref="TextMessage.VideoFrames"/>). Only
/// MOSS-VL does today; its video input also needs ffmpeg unless the caller supplies pre-decoded frames.</param>
/// <param name="RealtimeSession">Whether <see cref="ITextService.OpenRealtimeAsync"/> serves the model.</param>
public readonly record struct TextInputCapabilities(bool Images, bool Video, bool RealtimeSession)
{
    /// <summary>A model that takes text only, or a path that is not a known text model.</summary>
    public static TextInputCapabilities None => default;
}
