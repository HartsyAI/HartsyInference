using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;
using HartsyInference.World.Pipelines;

namespace HartsyInference.Engine.Services;

/// <summary>A world session over DIAMOND. Unlike <see cref="OasisWorldSession"/>, this is a genuinely per-frame-interactive loop: each queued action produces exactly one real <see cref="DiamondWorldPipeline.GenerateNextFrame"/> call (a full EDM sample), not a fixed-size batch. <see cref="SendAction"/> queues one action; <see cref="StreamAsync"/> drains the queue one action at a time, rolling the conditioning history forward after each frame (mirrors <c>DiamondGenPerfTests.Roll</c>). The session owns its history/seed state; the pipeline stays owned by the service that handed it over.</summary>
public sealed unsafe class DiamondWorldSession : IWorldSession
{
    private readonly DiamondWorldPipeline _pipeline;
    private readonly ConcurrentQueue<int> _actions = new();
    private readonly int? _seed;
    private Tensor _history;
    private int[] _actionHistory;
    private int _emitted;
    private bool _disposed;

    /// <summary>Creates a session seeded by <paramref name="request"/>'s first frame, replicated across the conditioning window (DIAMOND has no shorter real history to bootstrap from one image).</summary>
    internal DiamondWorldSession(DiamondWorldPipeline pipeline, WorldRequest request)
    {
        ImageData init = request.InitImage
            ?? throw new ArgumentException("DIAMOND rolls out from a first frame.", nameof(request));
        int size = pipeline.Config.ImgSize;
        if (init.Width != size || init.Height != size)
        {
            throw new ArgumentException(
                $"DIAMOND (Atari) needs a {size}x{size} seed frame; got {init.Width}x{init.Height}.", nameof(request));
        }
        _pipeline = pipeline;
        _history = pipeline.EncodeInitialHistory(init.Rgb);
        _actionHistory = new int[pipeline.Config.NumStepsConditioning];
        _seed = request.Seed < 0 ? null : (int)request.Seed;
    }

    /// <inheritdoc/>
    public void SendAction(string action)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _actions.Enqueue(ParseAction(action, _pipeline.Config.NumActions));
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<VideoFrame> StreamAsync([EnumeratorCancellation] CancellationToken cancel)
    {
        while (!_disposed && _actions.TryDequeue(out int action))
        {
            cancel.ThrowIfCancellationRequested();
            RollAction(action);
            int size = _pipeline.Config.ImgSize;
            int seed = _seed ?? Environment.TickCount + _emitted;
            Tensor next = await Task.Run(() => _pipeline.GenerateNextFrame(_history, _actionHistory, seed), cancel).ConfigureAwait(false);
            RollHistory(next);
            byte[] rgb = _pipeline.DecodeFrame(next);
            next.Dispose();
            yield return new VideoFrame { Rgb = rgb, Width = size, Height = size, Index = _emitted++ };
        }
    }

    /// <summary>Shifts the action-history window and appends the new action (mirrors <see cref="RollHistory"/>).</summary>
    private void RollAction(int action)
    {
        Array.Copy(_actionHistory, 1, _actionHistory, 0, _actionHistory.Length - 1);
        _actionHistory[^1] = action;
    }

    /// <summary>Drops the oldest frame from <see cref="_history"/> and appends <paramref name="next"/>.</summary>
    private void RollHistory(Tensor next)
    {
        int c = _pipeline.Config.ImgChannels, k = _pipeline.Config.NumStepsConditioning, size = _pipeline.Config.ImgSize;
        long frameLen = (long)c * size * size;
        float* dst = (float*)_history.DataPointer;
        Buffer.MemoryCopy(dst + frameLen, dst, (long)c * (k - 1) * size * size * sizeof(float), (long)c * (k - 1) * size * size * sizeof(float));
        Buffer.MemoryCopy((float*)next.DataPointer, dst + (long)c * (k - 1) * size * size, frameLen * sizeof(float), frameLen * sizeof(float));
    }

    /// <summary>Parses an action token into a discrete Atari action id: a raw integer in range, one of the standard Breakout names (noop/fire/left/right), or a no-op (with a warning) for anything else.</summary>
    private static int ParseAction(string action, int numActions)
    {
        string a = (action ?? "").Trim();
        if (int.TryParse(a, out int id) && id >= 0 && id < numActions) return id;
        int mapped = a.ToLowerInvariant() switch
        {
            "noop" or "none" or "" => 0,
            "fire" => 1,
            "right" => 2,
            "left" => 3,
            _ => -1,
        };
        if (mapped >= 0 && mapped < numActions) return mapped;
        Core.Logging.Logs.Warning($"[World] Ignoring unknown DIAMOND action '{action}' (treated as noop).");
        return 0;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _actions.Clear();
        _history.Dispose();
    }
}
