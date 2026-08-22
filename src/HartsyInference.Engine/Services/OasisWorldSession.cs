using System.Collections.Concurrent;
using System.Globalization;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;
using HartsyInference.World.ActionEncoders;
using HartsyInference.World.Pipelines;

namespace HartsyInference.Engine.Services;

/// <summary>A world session over Oasis. <b>The underlying pipeline is not resumable</b>: <c>OasisPipeline.Generate</c> takes a first frame plus a complete per-frame action plan and returns the whole rollout in one call, so this is not a true frame-by-frame interactive loop. What the session does instead is honest batching — <see cref="SendAction"/> queues actions, and <see cref="StreamAsync"/> drains everything queued so far into one rollout, yields its frames, then re-seeds the next batch from the last frame it produced. Actions queued while a batch is running are picked up by the next batch; the DiT's temporal context does <b>not</b> carry across batches, and the stream ends as soon as a drain finds the queue empty. Sending every action before enumerating (the CLI's pattern) therefore produces exactly one rollout, identical to the one-shot behavior.</summary>
public sealed class OasisWorldSession : IWorldSession
{
    /// <summary>Denoising steps per frame when the request does not set one.</summary>
    private const int DefaultDdimSteps = 10;

    /// <summary>Sentinel <see cref="KeyIndex"/> result for an explicit "do nothing this frame" token.</summary>
    private const int NoOpKey = -2;

    private readonly OasisPipeline _pipeline;
    private readonly ConcurrentQueue<float[]> _actions = new();
    private readonly int _steps;
    private readonly int? _seed;
    private byte[] _frameRgb;
    private int _width;
    private int _height;
    private int _emitted;
    private bool _disposed;

    /// <summary>Creates a session seeded by <paramref name="request"/>'s first frame; the pipeline stays owned by the service that handed it over.</summary>
    internal OasisWorldSession(OasisPipeline pipeline, WorldRequest request)
    {
        ImageData init = request.InitImage
            ?? throw new ArgumentException("Oasis rolls out from a first frame.", nameof(request));
        _pipeline = pipeline;
        _frameRgb = init.Rgb;
        _width = init.Width;
        _height = init.Height;
        _steps = request.Steps > 0 ? request.Steps : DefaultDdimSteps;
        _seed = request.Seed < 0 ? null : (int)request.Seed;
    }

    /// <inheritdoc/>
    public void SendAction(string action)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _actions.Enqueue(ParseAction(action));
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<VideoFrame> StreamAsync([EnumeratorCancellation] CancellationToken cancel)
    {
        while (!_disposed)
        {
            cancel.ThrowIfCancellationRequested();
            List<float[]> plan = new List<float[]>();
            while (_actions.TryDequeue(out float[]? action))
            {
                plan.Add(action);
            }
            if (plan.Count == 0)
            {
                yield break;
            }

            byte[] seedFrame = _frameRgb;
            int width = _width;
            int height = _height;
            bool first = _emitted == 0;
            (byte[][] frames, int outWidth, int outHeight, int _) = await Task.Run(
                () => _pipeline.Generate(
                    seedFrame, width, height, plan.ToArray(), plan.Count + 1, ddimSteps: _steps, seed: _seed,
                    onProgress: _ => cancel.ThrowIfCancellationRequested()),
                cancel).ConfigureAwait(false);

            _frameRgb = frames[^1];
            _width = outWidth;
            _height = outHeight;

            // Frame 0 of a batch is the seed frame the batch started from: emit it only for the very first batch.
            for (int i = first ? 0 : 1; i < frames.Length; i++)
            {
                cancel.ThrowIfCancellationRequested();
                yield return new VideoFrame { Rgb = frames[i], Width = outWidth, Height = outHeight, Index = _emitted++ };
            }
        }
    }

    /// <summary>Parses an action token into a 25-dim VPT row: <c>+</c>-separated key names (forward, back, left, right, jump, sneak, sprint, attack, use, drop, inventory) plus an optional <c>camera:x,y</c> term in [-1, 1]. An empty or unrecognized token becomes a no-op row (logged), never an exception mid-rollout.</summary>
    private static float[] ParseAction(string action)
    {
        byte[] keys = new byte[OasisActionEncoder.KeyCount];
        float cameraX = 0f;
        float cameraY = 0f;
        foreach (string term in (action ?? "").Split('+', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
        {
            if (term.StartsWith("camera:", StringComparison.OrdinalIgnoreCase))
            {
                string[] parts = term[7..].Split(',', StringSplitOptions.TrimEntries);
                if (parts.Length == 2
                    && float.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out float x)
                    && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float y))
                {
                    cameraX = x;
                    cameraY = y;
                    continue;
                }
                Logs.Warning($"[World] Ignoring malformed camera action '{term}' (expected camera:x,y).");
                continue;
            }
            int index = KeyIndex(term);
            if (index == NoOpKey)
            {
                continue;
            }
            if (index < 0)
            {
                Logs.Warning($"[World] Ignoring unknown action '{term}'.");
                continue;
            }
            keys[index] = 1;
        }
        float[] row = new float[OasisActionEncoder.ActionDim];
        OasisActionEncoder.BuildRow(keys, cameraX, cameraY, row);
        return row;
    }

    /// <summary>Index into the 23 VPT binary keys (ACTION_KEYS order), or -1 when the name is not one of them.</summary>
    private static int KeyIndex(string name) => name.ToLowerInvariant() switch
    {
        "inventory" => 0,
        "esc" => 1,
        "forward" => 11,
        "back" or "backward" => 12,
        "left" => 13,
        "right" => 14,
        "jump" => 15,
        "sneak" => 16,
        "sprint" => 17,
        "swaphands" => 18,
        "attack" => 19,
        "use" => 20,
        "pickitem" => 21,
        "drop" => 22,
        "none" or "noop" => NoOpKey,
        _ => -1,
    };

    /// <inheritdoc/>
    public void Dispose()
    {
        _disposed = true;
        _actions.Clear();
    }
}
