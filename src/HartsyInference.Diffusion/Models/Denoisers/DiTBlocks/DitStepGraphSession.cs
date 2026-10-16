using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>The capture/replay state machine for a denoise step whose op sequence is fixed for a given signature
/// (resolution, prompt, branch set): warm-up calls run eagerly so casts, uploads and promotions settle outside the
/// capture, call <see cref="CaptureCall"/> records the step through <see cref="IBackend.StepGraphBegin"/>, and later calls
/// replay it with one launch. The step must read and write only fixed device buffers its owner refreshes with
/// <see cref="IBackend.CopyInto"/> before each call. A capture or instantiation failure re-runs the step eagerly so the
/// generation stays correct, and <see cref="StepGraphFailureBudget"/> failures end capture for the session. The
/// backend's step-graph slot is shared between models, so losing ownership forces a re-warm.</summary>
/// <param name="label">Model name for the log lines.</param>
public sealed class DitStepGraphSession(string label)
{
    /// <summary>The call at a signature that records the graph; earlier calls warm up eagerly.</summary>
    public const int CaptureCall = 3;

    /// <summary>Signature changes tolerated before capture is abandoned as never converging.</summary>
    public const int MaxSignatureFlips = 8;

    private readonly StepGraphFailureBudget _failures = new();
    private long _signature = long.MinValue;
    private int _calls;
    private int _flips;

    /// <summary>True once capture has been abandoned for this session; every step then runs eagerly.</summary>
    public bool Dead { get; private set; }

    /// <summary>Runs <paramref name="step"/> under signature <paramref name="signature"/>: eagerly while warming up or
    /// dead, recorded on the capture call, replayed after that.</summary>
    public void Run(IBackend backend, object owner, long signature, Action step)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(owner);
        ArgumentNullException.ThrowIfNull(step);
        if (Dead)
        {
            step();
            return;
        }
        bool ownerLost = !ReferenceEquals(backend.StepGraphOwner, owner);
        if (signature != _signature || ownerLost)
        {
            backend.StepGraphReset();
            if (!ownerLost && _signature != long.MinValue && ++_flips > MaxSignatureFlips)
            {
                Dead = true;
                Logs.Warning($"[{label} graph] the step signature keeps changing; step-graph capture disabled for this session.");
                step();
                return;
            }
            _signature = signature;
            _calls = 0;
            backend.StepGraphOwner = owner;
        }
        _calls++;
        if (_calls > CaptureCall && backend.StepGraphReady)
        {
            backend.StepGraphLaunch();
            return;
        }
        if (_calls != CaptureCall)
        {
            step();
            return;
        }
        backend.StepGraphBegin();
        try
        {
            step();
        }
        catch (Exception ex)
        {
            Fail(backend, $"capture invalidated, running eagerly: {ex.Message}");
            step();
            return;
        }
        try
        {
            // Capture records without executing; this launch runs the step.
            backend.StepGraphEndAndLaunch();
            Logs.Info($"[{label} graph] denoise step captured; replaying with one graph launch per step.");
        }
        catch (Exception ex)
        {
            Fail(backend, $"graph instantiation failed, running eagerly: {ex.Message}");
            step();
        }
    }

    /// <summary>Drops the captured graph. Required before the owner's weights are freed or its fixed buffers replaced:
    /// a graph bakes device addresses, and replaying it over freed memory faults the whole context.</summary>
    public void Invalidate(IBackend backend, object owner)
    {
        ArgumentNullException.ThrowIfNull(backend);
        backend.StepGraphReset();
        if (ReferenceEquals(backend.StepGraphOwner, owner))
            backend.StepGraphOwner = null;
        _signature = long.MinValue;
        _calls = 0;
    }

    private void Fail(IBackend backend, string message)
    {
        backend.StepGraphReset();
        Dead = _failures.RecordFailure();
        _signature = long.MinValue;
        _calls = 0;
        Logs.Warning($"[{label} graph] {message}");
    }
}
