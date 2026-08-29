using HartsyInference.Engine;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>Wire representation of an engine progress tick for native and compatibility SSE routes.</summary>
internal sealed class StepPreviewPayload
{
    /// <summary>Completed/current step.</summary>
    public int Step { get; init; }

    /// <summary>Total scheduled steps.</summary>
    public int Total { get; init; }

    /// <summary>Static preview as a base64 PNG, or null when this tick has no preview.</summary>
    public string? PreviewPng { get; init; }

    /// <summary>Temporal preview frames as base64 PNGs in display order, or null for static previews.</summary>
    public string[]? PreviewFramesPng { get; init; }

    /// <summary>Preview width in pixels.</summary>
    public int PreviewWidth { get; init; }

    /// <summary>Preview height in pixels.</summary>
    public int PreviewHeight { get; init; }

    /// <summary>Builds a transport-safe payload by encoding the engine's raw RGB24 buffers.</summary>
    public static StepPreviewPayload Create(StepPreview preview)
    {
        int width = preview.PreviewWidth;
        int height = preview.PreviewHeight;
        int expectedLength = width > 0 && height > 0 ? width * height * 3 : 0;
        string? staticPng = preview.PreviewRgb is { } rgb && rgb.Length == expectedLength
            ? Convert.ToBase64String(PngEncoder.Encode(rgb, width, height))
            : null;
        string[]? framePngs = null;
        if (expectedLength > 0 && preview.PreviewFramesRgb is { Length: > 0 } frames &&
            frames.All(frame => frame is not null && frame.Length == expectedLength))
        {
            framePngs = new string[frames.Length];
            for (int i = 0; i < frames.Length; i++)
            {
                framePngs[i] = Convert.ToBase64String(PngEncoder.Encode(frames[i], width, height));
            }
        }

        return new StepPreviewPayload
        {
            Step = preview.Step,
            Total = preview.TotalSteps,
            PreviewPng = staticPng,
            PreviewFramesPng = framePngs,
            PreviewWidth = staticPng is null && framePngs is null ? 0 : width,
            PreviewHeight = staticPng is null && framePngs is null ? 0 : height,
        };
    }
}
