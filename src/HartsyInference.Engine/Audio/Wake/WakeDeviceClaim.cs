namespace HartsyInference.Engine.Audio.Wake;

/// <summary>One frame of claimed audio; see <see cref="WakeDeviceClaim.OnFrame"/>. A named delegate rather than
/// <c>Action&lt;ReadOnlySpan&lt;float&gt;&gt;</c> because a ref struct cannot be a generic type argument on
/// net8.0, which this package also targets.</summary>
public delegate void WakeInboundFrameHandler(ReadOnlySpan<float> samples);

/// <summary>A host's claim on one connected satellite's turns, from <see cref="WakeService.Claim"/>.
///
/// <para>While a device carries one of these (<see cref="WakeSession.Claim"/>), the wake worker delivers that
/// device's decoded inbound audio to <see cref="OnFrame"/> instead of running its own wake scoring,
/// end-of-speech capture and transcription for it — the device is still connected, still pinged, and its
/// outbound audio path (<see cref="WakeService.BeginAudio"/>/<see cref="WakeService.SendAudioAsync"/>) still
/// works exactly as before; only inbound routing changes.</para>
///
/// <para>Plain data, not <see cref="IDisposable"/>: release through <see cref="WakeService.Release"/>, passing
/// this same instance back, which is public so a host can hold the reference it was given and so a test can
/// construct one directly and drive <see cref="WakeSession.Claim"/> without a live <see cref="WakeService"/> at
/// all — the same reasoning <see cref="WakeAudioStream"/>'s own public constructor documents.</para></summary>
public sealed class WakeDeviceClaim
{
    /// <summary>Called on the wake worker thread with each frame of 16 kHz mono float audio for the claimed
    /// device, post-denoise when noise suppression is on, normalized to [-1, 1] — the engine's wake path itself
    /// carries int16-scaled audio internally, and this is the one boundary that converts it to the scale every
    /// other audio consumer in the codebase expects. Must not block or throw past what it can recover from: it
    /// runs on the same thread that drains every other connected device, claimed or not.</summary>
    public WakeInboundFrameHandler OnFrame { get; }

    /// <summary>Called at most once, if the device disconnects while this claim is still in effect — after the
    /// claim has already been cleared, so the host does not need to poll to find out its turn ended out from
    /// under it. Not called by an ordinary <see cref="WakeService.Release"/>, and not called at all if the
    /// claim was already released before the disconnect.</summary>
    public Action? OnDisconnected { get; }

    public WakeDeviceClaim(WakeInboundFrameHandler onFrame, Action? onDisconnected = null)
    {
        OnFrame = onFrame ?? throw new ArgumentNullException(nameof(onFrame));
        OnDisconnected = onDisconnected;
    }
}
