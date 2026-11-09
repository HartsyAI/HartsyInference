using System.Diagnostics;
using System.Globalization;
using System.Text;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Imaging;
using HartsyInference.Engine.Services;
using HartsyInference.Video.Encoding;
using Spectre.Console;

namespace HartsyInference.Cli.Dispatch;

/// <summary><c>hartsy text --realtime</c>: replays a video file or a frame folder into a live vision-language session at its own pace,
/// takes further prompts from stdin (one per line), and prints the model's reply as it streams.</summary>
internal static class RealtimeDispatch
{
    /// <summary>Runs one live session until its frames are spent and the model has been silent for <c>idle-exit</c> seconds, stdin
    /// closes after that, or <paramref name="cancel"/> fires (Ctrl+C).</summary>
    public static async Task<GeneratedArtifact> RunAsync(IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters,
        bool quiet, CancellationToken cancel)
    {
        float? temperature = parameters.GetFloatOrNull("temperature") is float t && t > 0 ? t : null;
        RealtimeVlmRequest request = new()
        {
            Device = parameters.GetStringOrNull("device"),
            SystemPrompt = parameters.GetStringOrNull("system"),
            InitialPrompt = prompt,
            MaxTokensPerSecond = parameters.GetDoubleOrNull("tokens-per-second"),
            Temperature = temperature,
            TopP = temperature is null ? null : parameters.GetFloatOrNull("top-p"),
            Seed = parameters.GetInt("seed", -1) is int seed and >= 0 ? (ulong)seed : null,
            LowVramQuant = parameters.GetBool("low-vram-quant", false) ? "on" : null,
            WeightPrecision = parameters.GetStringOrNull("precision"),
        };
        TimeSpan idleExit = TimeSpan.FromSeconds(parameters.GetDoubleOrNull("idle-exit") ?? 3.0);
        bool showSilence = parameters.GetBool("show-silence", false);

        using CancellationTokenSource stop = CancellationTokenSource.CreateLinkedTokenSource(cancel);
        IRealtimeVlmSession session = await engine.Text.OpenRealtimeAsync(spec, request, cancel).ConfigureAwait(false);
        StringBuilder reply = new();
        Stopwatch sinceOutput = Stopwatch.StartNew();
        bool lastWasSilence = false, framesDone = false;
        try
        {
            Task feeder = Task.Run(async () =>
            {
                await foreach (TimedRgbFrame frame in LiveFrames(parameters, stop.Token).ConfigureAwait(false))
                {
                    if (session.PushFrame(new ImageData { Rgb = frame.Frame.Rgb.ToArray(), Width = frame.Frame.Width, Height = frame.Frame.Height },
                            frame.TimestampSeconds) && !quiet)
                        Console.Error.WriteLine($"[frame queue full: dropped the oldest frame at {frame.TimestampSeconds:F1}s]");
                }
                Volatile.Write(ref framesDone, true);
            }, stop.Token);
            _ = Task.Run(() => ReadPrompts(session, stop.Token), stop.Token);
            Task watchdog = Task.Run(async () =>
            {
                // Ends the run once every frame is in and the model has stayed silent for the idle window.
                while (!stop.IsCancellationRequested)
                {
                    await Task.Delay(100, stop.Token).ConfigureAwait(false);
                    if (Volatile.Read(ref framesDone) && Volatile.Read(ref lastWasSilence) && sinceOutput.Elapsed > idleExit)
                    {
                        stop.Cancel();
                        return;
                    }
                }
            }, stop.Token);

            try
            {
                await foreach (RealtimeVlmChunk chunk in session.StreamAsync(stop.Token).ConfigureAwait(false))
                {
                    sinceOutput.Restart();
                    Volatile.Write(ref lastWasSilence, chunk.Kind == RealtimeVlmChunkKind.Silence);
                    switch (chunk.Kind)
                    {
                        case RealtimeVlmChunkKind.Text:
                            reply.Append(chunk.Text);
                            Console.Write(chunk.Text);
                            break;
                        case RealtimeVlmChunkKind.Silence when showSilence:
                            Console.Write(" [silence] ");
                            break;
                        case RealtimeVlmChunkKind.Control when showSilence:
                            Console.Write($" [{chunk.Text.Trim('<', '>', '|')}] ");
                            break;
                        case RealtimeVlmChunkKind.RoundStart:
                            Console.WriteLine();
                            if (!quiet) AnsiConsole.MarkupLine($"[grey][[round]] {Markup.Escape(chunk.Text)}[/]");
                            reply.Append('\n');
                            break;
                        case RealtimeVlmChunkKind.FramesDropped when !quiet:
                            Console.Error.WriteLine($"[{chunk.Count} frame(s) dropped: the model is slower than the frame rate]");
                            break;
                        case RealtimeVlmChunkKind.Error:
                            throw new InvalidOperationException($"The live session failed: {chunk.Text}");
                    }
                }
            }
            catch (OperationCanceledException) when (stop.IsCancellationRequested && !cancel.IsCancellationRequested)
            {
                // The watchdog ended the run.
            }
            Console.WriteLine();
            await feeder.ContinueWith(static _ => { }, TaskScheduler.Default).ConfigureAwait(false);
            await watchdog.ContinueWith(static _ => { }, TaskScheduler.Default).ConfigureAwait(false);
        }
        finally
        {
            stop.Cancel();
            RealtimeVlmSessionStats stats = session.Stats;
            await session.DisposeAsync().ConfigureAwait(false);
            if (!quiet)
                AnsiConsole.MarkupLine($"[grey]Live session: {stats.GeneratedTokens} tokens, {stats.FramesProcessed} frames "
                    + $"({stats.FramesDropped} dropped), {stats.TextTokens} text and {stats.VisionTokens} vision tokens held.[/]");
        }

        GeneratedArtifact artifact = new() { Kind = ArtifactKind.Text, Text = reply.ToString(), Extension = "txt", Streamed = true };
        artifact.Meta["mode"] = "realtime";
        return artifact;
    }

    /// <summary>The run's frames at their own pace: a video file sampled at <c>frame-fps</c> (default 1), or a folder's images, frame
    /// i at i / <c>frame-fps</c> seconds.</summary>
    private static async IAsyncEnumerable<TimedRgbFrame> LiveFrames(ParamState parameters,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancel)
    {
        double fps = parameters.GetDoubleOrNull("frame-fps") ?? parameters.GetDoubleOrNull("video-fps") ?? 1.0;
        if (parameters.GetStringOrNull("video") is { } video)
        {
            // Every planned frame is kept: a live replay has no clip budget, only the session's own vision cap.
            VideoFrameSamplingOptions sampling = new(Fps: fps, MaxFrames: int.MaxValue, VideoMaxPixels: long.MaxValue);
            using FfmpegFrameSource source = new(video, sampling, paced: true);
            await foreach (TimedRgbFrame frame in source.ReadAsync(cancel).ConfigureAwait(false))
                yield return frame;
            yield break;
        }
        if (parameters.GetStringOrNull("frames-dir") is { } folder)
        {
            Stopwatch clock = Stopwatch.StartNew();
            foreach (Engine.Requests.TimedImage image in GenerationDispatch.FolderFrames(folder, fps))
            {
                TimeSpan due = TimeSpan.FromSeconds(image.TimestampSeconds) - clock.Elapsed;
                if (due > TimeSpan.Zero) await Task.Delay(due, cancel).ConfigureAwait(false);
                yield return new TimedRgbFrame(new RawRgbFrame(image.Image.Rgb, image.Image.Width, image.Image.Height), image.TimestampSeconds);
            }
        }
    }

    /// <summary>Feeds each stdin line to the session as a prompt until stdin closes.</summary>
    private static async Task ReadPrompts(IRealtimeVlmSession session, CancellationToken cancel)
    {
        try
        {
            while (!cancel.IsCancellationRequested)
            {
                string? line = await Console.In.ReadLineAsync(cancel).ConfigureAwait(false);
                if (line is null) return;
                if (line.Trim().Length > 0) session.PushPrompt(line.Trim());
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (ObjectDisposedException)
        {
            // The session ended while a line was pending.
        }
    }
}
