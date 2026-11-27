using System.Runtime.CompilerServices;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.WorldViewer;

/// <summary>The frames of a steered session. An OnChange session is enumerated once. A PerStep session takes one action per step, so
/// the stream sends the current action before each enumeration and enumerates again when the session ends its run: a session such as
/// DIAMOND or Oasis ends whenever its action queue drains.</summary>
public sealed class WorldFrameStream
{
    private readonly IWorldSession _session;
    private readonly WorldSteering _steering;
    private readonly int _maxFrames;

    /// <summary>Streams <paramref name="session"/> through <paramref name="steering"/>. <paramref name="maxFrames"/> ends the stream after
    /// that many frames; 0 runs until the session or the caller ends it.</summary>
    public WorldFrameStream(IWorldSession session, WorldSteering steering, int maxFrames)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(steering);
        ArgumentOutOfRangeException.ThrowIfNegative(maxFrames);
        _session = session;
        _steering = steering;
        _maxFrames = maxFrames;
    }

    /// <summary>Yields the session's frames in order, up to the frame limit, and sends the steering's actions as the cadence asks.</summary>
    public async IAsyncEnumerable<VideoFrame> StreamAsync([EnumeratorCancellation] CancellationToken cancel)
    {
        bool perStep = _steering.Scheme?.Cadence == WorldControlCadence.PerStep;
        int yielded = 0;
        while (true)
        {
            cancel.ThrowIfCancellationRequested();
            if (perStep)
            {
                _session.SendAction(_steering.CurrentAction());
            }
            int delivered = 0;
            await using IAsyncEnumerator<VideoFrame> frames = _session.StreamAsync(cancel).GetAsyncEnumerator(cancel);
            while (true)
            {
                if (!await frames.MoveNextAsync().ConfigureAwait(false))
                {
                    break;
                }
                delivered++;
                yielded++;
                yield return frames.Current;
                if (_maxFrames > 0 && yielded >= _maxFrames)
                {
                    yield break;
                }
            }
            if (!perStep)
            {
                yield break;
            }
            if (delivered == 0)
            {
                throw new InvalidOperationException("The session ended a step without a frame for its action.");
            }
        }
    }
}
