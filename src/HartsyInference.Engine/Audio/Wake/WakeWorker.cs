using System.Collections.Concurrent;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Cpu;

namespace HartsyInference.Engine.Audio.Wake;

/// <summary>The always-on detection loop: one thread that drains every connected device and scores its audio.
///
/// <para>This is the engine's first genuinely continuous worker, so it sets a pattern worth stating explicitly.
/// It owns a <b>private <see cref="CpuBackend"/></b> and never touches <c>AudioRuntime</c>'s generation lock or
/// an <c>InferenceQueue</c> slot. Those are serialized at concurrency 1 for good reason, and a listener that
/// wakes every 80 ms forever would hold them often enough to starve every image, video and speech request on
/// the engine. Wake scoring is a few milliseconds of small convolutions, so it does not need — and must not
/// take — the shared inference path.</para>
///
/// <para>Everything expensive that a detection triggers (transcription, speaker identification) is burst work
/// and is dispatched off this thread through the normal queue, so a slow Whisper pass never stalls detection
/// for other devices.</para></summary>
public sealed class WakeWorker : IDisposable
{
    private const int DrainBufferSamples = 8192;
    /// <summary>Headroom for the denoiser to emit a frame it had been holding, on top of what was drained.</summary>
    private const int DenoiseSlackSamples = 1024;
    private const int IdleSleepMs = 10;

    private readonly ConcurrentDictionary<string, WakeSession> _sessions;
    private readonly Func<WakeSession, WakeDetection, Task> _onDetection;
    private readonly CancellationTokenSource _stopping = new();
    private readonly Thread _thread;
    private int _disposed;

    /// <summary>Detection steps executed since start; a health signal for "is the loop actually running".</summary>
    public long StepsProcessed { get; private set; }

    public WakeWorker(ConcurrentDictionary<string, WakeSession> sessions, Func<WakeSession, WakeDetection, Task> onDetection)
    {
        _sessions = sessions ?? throw new ArgumentNullException(nameof(sessions));
        _onDetection = onDetection ?? throw new ArgumentNullException(nameof(onDetection));
        _thread = new Thread(Run)
        {
            IsBackground = true,
            Name = "hartsy-wake-worker",
            // Above normal so a busy GPU generation queue cannot starve the listener into dropping audio.
            Priority = ThreadPriority.AboveNormal,
        };
    }

    public void Start() => _thread.Start();

    private void Run()
    {
        float[] buffer = new float[DrainBufferSamples];
        float[] denoised = new float[DrainBufferSamples + DenoiseSlackSamples];
        // Only filled, and only converted to [-1, 1] scale, for a claimed device's frames -- an unclaimed one
        // never touches this, so it costs nothing beyond its own allocation for the common case.
        float[] normalized = new float[DrainBufferSamples + DenoiseSlackSamples];
        List<WakeDetection> detections = [];
        // Private to this thread: the shared audio runtime serializes on one generation lock, and an 80 ms
        // cadence would hold it often enough to starve every other request on the engine.
        using IBackend backend = new CpuBackend();
        try
        {
            Logs.Info("[Audio][Wake] Detection worker started.");

            while (!_stopping.IsCancellationRequested)
            {
                bool didWork = false;
                foreach (WakeSession session in _sessions.Values)
                {
                    if (session.State == WakeSessionState.Handshake) continue;
                    if (!session.PendingWords.IsEmpty) session.ApplyPendingWords();
                    if (session.RequestReset)
                    {
                        session.Pipeline.Reset();
                        session.Denoiser?.Reset();
                        session.ResetVad();
                        session.RequestReset = false;
                    }

                    int read = session.Drain(buffer);
                    if (read == 0) continue;
                    didWork = true;

                    // Snapshotted once per device per iteration: a claim or release landing mid-iteration is
                    // fine either way (this frame goes wherever the snapshot says, the next one sees whatever
                    // is current then), and the worker must never block waiting for a host that is slow to
                    // claim or release.
                    WakeDeviceClaim? claim = session.Claim;
                    try
                    {
                        // Denoise before scoring (or before handing off to a claim), and only for that: the
                        // session's capture buffer keeps the raw audio, so transcription and speaker
                        // identification still see what the microphone actually heard. The denoiser holds
                        // audio back while it fills, so `processed` is not `read` and the pipeline (or the
                        // claim) must be given the returned count.
                        ReadOnlySpan<float> toProcess;
                        if (session.Denoiser is null) toProcess = buffer.AsSpan(0, read);
                        else
                        {
                            int processed = session.Denoiser.Process(backend, buffer.AsSpan(0, read), denoised);
                            toProcess = denoised.AsSpan(0, processed);
                        }
                        // Pipeline.Push clears `detections` itself before adding its own results, but the
                        // claimed branch below never calls it -- without an explicit clear here, a claimed
                        // device's iteration would still see whatever an earlier, unclaimed device's Push left
                        // in the shared list this loop reuses across every session, and dispatch a detection
                        // that belongs to a different device entirely.
                        detections.Clear();
                        if (!toProcess.IsEmpty)
                        {
                            if (claim is not null)
                            {
                                // A host session owns this device's turns now: hand it the audio instead of
                                // scoring or end-of-speech-tracking it. The wake path carries int16-scaled
                                // audio internally (see PushVad's own division below); every other audio
                                // consumer in the codebase expects [-1, 1], so this boundary converts once
                                // rather than leaking the internal scale to a claim's own caller.
                                for (int i = 0; i < toProcess.Length; i++) normalized[i] = toProcess[i] / 32768f;
                                try
                                {
                                    claim.OnFrame(normalized.AsSpan(0, toProcess.Length));
                                }
                                catch (Exception ex)
                                {
                                    // Caught separately from the outer catch below: a throwing OnFrame is the
                                    // host's bug, not this device's audio state, so resetting the pipeline,
                                    // denoiser and VAD here would be pointless (a claimed device never reads
                                    // them) and a host that throws persistently would otherwise log a reset
                                    // error roughly every 80 ms.
                                    Logs.Error($"[Audio][Wake] WakeDeviceClaim.OnFrame threw for '{session.DeviceId}'.", ex);
                                }
                            }
                            else
                            {
                                session.Pipeline.Push(backend, toProcess, detections);
                                StepsProcessed += toProcess.Length / WakeDetectionPipeline.ChunkSamples;
                                // End-of-speech runs on the denoised audio for the same reason scoring does: it
                                // is deciding whether a person is talking, and room noise is exactly what would
                                // keep it from ever hearing the pause.
                                session.PushVad(backend, toProcess);
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        // One device's bad state must not take the loop down for every other device. Resetting
                        // the pipeline/denoiser/VAD here is harmless even for a claimed device that never uses
                        // them while claimed: it only clears state nothing is currently reading.
                        Logs.Error($"[Audio][Wake] Detection failed for device '{session.DeviceId}'; resetting it.", ex);
                        session.Pipeline.Reset();
                        session.Denoiser?.Reset();
                        session.ResetVad();
                        continue;
                    }

                    foreach (WakeDetection detection in detections)
                    {
                        Logs.Info($"[Audio][Wake] '{detection.Word}' detected on '{session.DeviceId}' (score {detection.Score:F3}).");
                        WakeSession captured = session;
                        WakeDetection value = detection;
                        _ = Task.Run(async () =>
                        {
                            try { await _onDetection(captured, value).ConfigureAwait(false); }
                            catch (Exception ex) { Logs.Error($"[Audio][Wake] Detection handler failed for '{captured.DeviceId}'.", ex); }
                        });
                    }
                }

                if (!didWork) Thread.Sleep(IdleSleepMs);
            }
        }
        catch (Exception ex)
        {
            Logs.Error("[Audio][Wake] Detection worker stopped unexpectedly; no device will produce detections until the service restarts.", ex);
        }
        finally
        {
            Logs.Info("[Audio][Wake] Detection worker stopped.");
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _stopping.Cancel();
        _thread.Join(TimeSpan.FromSeconds(5));
        _stopping.Dispose();
    }
}
