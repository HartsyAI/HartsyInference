using HartsyInference.Core.Logging;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Endpoints;

/// <summary>Native video route. <see cref="IVideoService"/> only exposes a frame stream (<see cref="IAsyncEnumerable{T}"/> of <see cref="VideoFrame"/>, no <c>Task&lt;VideoResult&gt;</c>) — there's no way to build a non-streaming variant without buffering an entire video's frames in server memory first. An H.264 muxer DOES exist (<c>HartsyInference.Video.Encoding.FfmpegProcessEncoder</c>, ffmpeg subprocess — the CLI's restore path uses it), but it isn't reachable from this project (API references Core + Engine only), so this ships SSE-streamed raw frames; a muxed-download variant is future work. Runs through the long-running queue (video generation can take minutes), not the fast one every other route uses.</summary>
public static class VideoEndpoints
{
    /// <summary>Maps <c>/v1/native/video/stream</c>.</summary>
    public static void MapVideoEndpoints(this WebApplication app)
    {
        app.MapPost("/v1/native/video/stream", async (
            NativeVideoRequest req, IInferenceEngine engine,
            [FromKeyedServices(QueueKeys.LongRunning)] InferenceQueue queue, HttpContext ctx, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Video);
            await SseHelpers.RunAsync(ctx, queue, async (writer, jsonOptions) =>
            {
                // Runs inside the queue's held slot — call the service directly, not through a second EnqueueAsync.
                Progress<StepPreview> progress = new Progress<StepPreview>(p =>
                    writer.TryWrite(SseHelpers.Event("progress", new { step = p.Step, total = p.TotalSteps }, jsonOptions)));
                VideoGenerationResult result = await engine.Video.GenerateAsync(spec, req.Request, progress, ct);
                foreach (VideoFrame frame in result.Frames)
                {
                    ct.ThrowIfCancellationRequested();
                    writer.TryWrite(SseHelpers.Event("frame", new
                    {
                        index = frame.Index,
                        png = Convert.ToBase64String(PngEncoder.Encode(frame.Rgb, frame.Width, frame.Height)),
                    }, jsonOptions));
                }
                if (result.Audio is not null && !result.Audio.IsEmpty)
                {
                    writer.TryWrite(SseHelpers.Event("audio", new
                    {
                        sampleRate = result.Audio.SampleRate,
                        channels = result.Audio.ChannelCount,
                        wav = Convert.ToBase64String(AudioClipCodec.EncodeWav(result.Audio)),
                    }, jsonOptions));
                }
                writer.TryWrite(SseHelpers.Event("complete", new
                {
                    frames = result.Frames.Count,
                    savedPath = Persist(req, result),
                }, jsonOptions));
            }, ct);
        });
    }

    /// <summary>Writes the frame sequence (and the soundtrack beside it) into a numbered directory under the output root, the same layout the CLI produces. Null when the request opted out or the write failed.</summary>
    private static string? Persist(NativeVideoRequest req, VideoGenerationResult result)
    {
        if (req.Save == false || result.Frames.Count == 0)
            return null;
        try
        {
            VideoFrame first = result.Frames[0];
            byte[][] rgb = new byte[result.Frames.Count][];
            for (int i = 0; i < result.Frames.Count; i++)
                rgb[i] = result.Frames[i].Rgb;
            return VideoOutputWriter.Write(rgb, first.Width, first.Height,
                OutputWriter.ResolveDir(req.OutputDir), req.Request.Prompt,
                result.Audio, result.Fps ?? 24).Directory;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[api] generated video could not be saved: {ex.Message}");
            return null;
        }
    }
}
