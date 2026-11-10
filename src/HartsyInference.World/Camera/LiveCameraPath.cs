namespace HartsyInference.World.Camera;

/// <summary>A camera path that grows while a world is generated: live input sets the controls being held or queues scripted
/// frames, and the generator asks for more poses a chunk at a time. Scripted frames play first, in order; once they run
/// out, the held controls apply to every further frame. Safe to drive from an input thread while a compute thread extends
/// the path.</summary>
/// <typeparam name="TControls">The controls held during one frame.</typeparam>
public sealed class LiveCameraPath<TControls>
{
    private readonly ICameraIntegrator<TControls> _integrator;
    private readonly object _gate = new();
    private readonly List<float[]> _poses = [];
    private readonly Queue<(TControls Controls, int Frames)> _scripted = new();
    private (TControls Controls, int Frames) _playing;
    private TControls _held;
    private int _queuedFrames;

    /// <summary>Starts the path at <paramref name="integrator"/>'s current pose with <paramref name="idle"/> held.</summary>
    public LiveCameraPath(ICameraIntegrator<TControls> integrator, TControls idle)
    {
        ArgumentNullException.ThrowIfNull(integrator);
        _integrator = integrator;
        _held = idle;
        _poses.Add(integrator.Pose);
    }

    /// <summary>Poses integrated so far, the starting pose first.</summary>
    public int PoseCount
    {
        get
        {
            lock (_gate)
            {
                return _poses.Count;
            }
        }
    }

    /// <summary>Scripted frames not yet integrated.</summary>
    public int QueuedFrames
    {
        get
        {
            lock (_gate)
            {
                return _queuedFrames;
            }
        }
    }

    /// <summary>Holds <paramref name="controls"/> from the next unscripted frame on, replacing what was held.</summary>
    public void Hold(TControls controls)
    {
        lock (_gate)
        {
            _held = controls;
        }
    }

    /// <summary>Queues <paramref name="frames"/> frames of <paramref name="controls"/> behind any already queued.</summary>
    public void Enqueue(TControls controls, int frames)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(frames, 1);
        lock (_gate)
        {
            _scripted.Enqueue((controls, frames));
            _queuedFrames += frames;
        }
    }

    /// <summary>Drops the scripted frames not yet integrated; the held controls are unchanged.</summary>
    public void ClearQueued()
    {
        lock (_gate)
        {
            _scripted.Clear();
            _playing = default;
            _queuedFrames = 0;
        }
    }

    /// <summary>Integrates frames until the path holds at least <paramref name="poseCount"/> poses, then returns a snapshot of
    /// the first <paramref name="poseCount"/>. The controls held when the call starts steer every unscripted frame it adds;
    /// later input steers the frames a later call adds.</summary>
    public IReadOnlyList<float[]> ExtendTo(int poseCount)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(poseCount, 1);
        lock (_gate)
        {
            while (_poses.Count < poseCount)
                _poses.Add(_integrator.Step(NextControls()));
            return _poses.GetRange(0, poseCount);
        }
    }

    // The controls of the next frame: the scripted segment being played while one remains, else the held controls.
    private TControls NextControls()
    {
        if (_playing.Frames == 0 && _scripted.Count > 0)
            _playing = _scripted.Dequeue();
        if (_playing.Frames == 0)
            return _held;
        _playing.Frames--;
        _queuedFrames--;
        return _playing.Controls;
    }
}
