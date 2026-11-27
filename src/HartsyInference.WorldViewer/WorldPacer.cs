namespace HartsyInference.WorldViewer;

/// <summary>Decides, from the clock and the buffered frame count, when the next frame is shown. Frames play at the stream's frame
/// rate. A clock that falls well behind is rebased rather than replayed as a burst.</summary>
public sealed class WorldPacer
{
    private readonly TimeSpan _period;
    private readonly int _startFrames;
    private readonly int _resumeFrames;
    private TimeSpan _nextDue;
    private bool _playing;
    private bool _started;

    /// <summary>Creates a pacer at <paramref name="fps"/>; it starts once <paramref name="startFrames"/> are buffered and resumes after an
    /// underrun once <paramref name="resumeFrames"/> are.</summary>
    public WorldPacer(int fps, int startFrames, int resumeFrames)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(fps, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(resumeFrames, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(startFrames, resumeFrames);
        _period = TimeSpan.FromTicks(TimeSpan.TicksPerSecond / fps);
        _startFrames = startFrames;
        _resumeFrames = resumeFrames;
    }

    /// <summary>Whether frames are playing now rather than waiting for the buffer.</summary>
    public bool Playing => _playing;

    /// <summary>Whether a frame has ever been shown.</summary>
    public bool Started => _started;

    /// <summary>Decides what to do at <paramref name="now"/> with <paramref name="buffered"/> frames waiting.</summary>
    public WorldPaceStep Step(TimeSpan now, int buffered, bool streamEnded)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(buffered);
        if (!_playing)
        {
            int needed = _started ? _resumeFrames : _startFrames;
            if (buffered < needed && !(streamEnded && buffered > 0))
            {
                return streamEnded ? WorldPaceStep.Finished : WorldPaceStep.Buffering;
            }
            _playing = true;
            _started = true;
            _nextDue = now;
        }
        if (now < _nextDue)
        {
            return WorldPaceStep.Hold;
        }
        if (buffered == 0)
        {
            if (streamEnded)
            {
                return WorldPaceStep.Finished;
            }
            _playing = false;
            return WorldPaceStep.Buffering;
        }
        _nextDue += _period;
        if (now - _nextDue > _period + _period)
        {
            _nextDue = now + _period;
        }
        return WorldPaceStep.Show;
    }
}
