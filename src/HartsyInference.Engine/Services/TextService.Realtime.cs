using System.Runtime.CompilerServices;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.Multimodal.Streaming;
using HartsyInference.LLM.Sampling;

namespace HartsyInference.Engine.Services;

/// <summary>Live vision-language sessions in the text service.</summary>
public sealed partial class TextService
{
    /// <inheritdoc/>
    /// <remarks>The session holds the slot's lock until it is disposed, so the slot's model cannot be replaced or unloaded under it; an
    /// unload waits for it as for any request and reports a timeout if the session outlives the wait. Each forward takes the device gate
    /// on its own, so the device's other services run between tokens.</remarks>
    public async Task<IRealtimeVlmSession> OpenRealtimeAsync(ModelSpec spec, RealtimeVlmRequest request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(request);
        string deviceKey = NormalizeDeviceKey(request.Device);
        TextDeviceSlot slot = _slots.GetOrAdd(deviceKey, static _ => new TextDeviceSlot());
        await slot.Lock.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            TextRequest load = new() { Messages = [], Device = request.Device, LowVramQuant = request.LowVramQuant };
            await Task.Run(() =>
            {
                if (slot.HasLeases && ReplacesLoadedModel(slot, spec.LocalPath) && !slot.WaitForLeases(TimeSpan.FromSeconds(UnloadWaitSeconds)))
                    throw new HartsyInferenceException($"Scheduled requests on '{slot.LoadedPath}' did not finish within {UnloadWaitSeconds}s; retry.");
                using IDisposable gate = DeviceGate.AcquireAllOrdinals(LoadGateOrdinalsFor(slot, deviceKey, spec.LocalPath), cancel);
                LoadInto(slot, deviceKey, spec, load);
            }, cancel).ConfigureAwait(false);
            if (slot.MossVl is not { } runtime)
                throw new NotSupportedException(
                    $"Live vision sessions are served for MOSS-VL-Realtime; '{spec.Requested}' is not a streaming vision-language model.");

            int[] ordinals = [.. GateOrdinalsFor(slot, deviceKey)];
            RealtimeVlmOptions options = new()
            {
                FrameQueueSize = request.FrameQueueSize,
                MaxTokensPerSecond = request.MaxTokensPerSecond ?? new RealtimeVlmOptions().MaxTokensPerSecond,
                MaxFramesPerCycle = request.MaxFramesPerCycle ?? 0,
                Sampling = request.Temperature is null && request.TopP is null && request.Seed is null
                    ? null
                    : new SamplingOptions
                    {
                        Temperature = request.Temperature ?? 1.0f,
                        TopP = request.TopP ?? 1.0f,
                        Seed = request.Seed ?? 0,
                    },
            };
            RealtimeVlmSession session = runtime.OpenRealtime(options, request.MaxVisionTokens,
                c => DeviceGate.AcquireAllOrdinals(ordinals, c));
            try
            {
                session.Start(request.SystemPrompt, request.InitialPrompt);
            }
            catch
            {
                await session.DisposeAsync().ConfigureAwait(false);
                throw;
            }
            Logs.Info($"[TextService] Opened a live MOSS-VL session on {deviceKey}.");
            return new LiveVlmSession(session, () => slot.Lock.Release());
        }
        catch
        {
            slot.Lock.Release();
            throw;
        }
    }

    /// <summary>The Engine face of a <see cref="RealtimeVlmSession"/>; disposing it stops the loop and releases the slot.</summary>
    private sealed class LiveVlmSession(RealtimeVlmSession inner, Action release) : IRealtimeVlmSession
    {
        private int _disposed;

        public RealtimeVlmSessionStats Stats
        {
            get
            {
                RealtimeVlmStats s = inner.Stats;
                return new RealtimeVlmSessionStats(s.TextTokens, s.VisionTokens, s.GeneratedTokens, s.FramesProcessed, s.FramesDropped);
            }
        }

        public bool PushFrame(ImageData frame, double timestampSeconds)
        {
            ArgumentNullException.ThrowIfNull(frame);
            return inner.PushFrame(new RawRgbFrame(frame.Rgb, frame.Width, frame.Height), timestampSeconds);
        }

        public void PushPrompt(string prompt) => inner.PushPrompt(prompt);

        public async IAsyncEnumerable<RealtimeVlmChunk> StreamAsync([EnumeratorCancellation] CancellationToken cancel = default)
        {
            await foreach (RealtimeVlmEvent e in inner.ReadAsync(cancel).ConfigureAwait(false))
            {
                yield return e.Kind switch
                {
                    RealtimeVlmEventKind.RoundStart => new RealtimeVlmChunk(RealtimeVlmChunkKind.RoundStart, e.Text),
                    RealtimeVlmEventKind.Text => new RealtimeVlmChunk(RealtimeVlmChunkKind.Text, e.Text),
                    RealtimeVlmEventKind.Silence => new RealtimeVlmChunk(RealtimeVlmChunkKind.Silence, e.Text),
                    RealtimeVlmEventKind.Control => new RealtimeVlmChunk(RealtimeVlmChunkKind.Control, e.Text),
                    RealtimeVlmEventKind.FramesDropped => new RealtimeVlmChunk(RealtimeVlmChunkKind.FramesDropped, Count: e.Count),
                    _ => new RealtimeVlmChunk(RealtimeVlmChunkKind.Error, e.Text),
                };
            }
        }

        public async ValueTask DisposeAsync()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            try
            {
                await inner.DisposeAsync().ConfigureAwait(false);
            }
            finally
            {
                release();
            }
        }
    }
}
