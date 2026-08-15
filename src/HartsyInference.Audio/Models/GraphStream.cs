using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models;

/// <summary>Per-stream CUDA-graph decode state for a transformer backbone: fixed-address input-embedding and
/// output-hidden buffers (refreshed/read OUTSIDE any capture), a device position buffer, a warmup counter, and the
/// captured single-frame graph once warm. The step for a steady-state frame (one appended row per stream) is
/// captured once and replayed per frame, collapsing the ~layers×kernels launches into one <c>cuGraphLaunch</c>.
/// <paramref name="rows"/> is 1 for a single stream and 2 for a position-aligned classifier-free pair sharing one
/// step. The capture bakes both buffer addresses AND the KV caches' addresses, so one instance belongs to exactly
/// one generation over one set of caches.</summary>
internal sealed class GraphStream : IDisposable
{
    private readonly IBackend _backend;
    public readonly Tensor InEmbed;    // [1,rows,h] — CopyInto'd from the frame's new embedding before each step
    public readonly Tensor OutHidden;  // [1,rows,h] — post-final-norm hidden, read via ReadResidentInto
    public readonly ulong DevicePos;
    public object? Graph;
    public int Warmed;                 // frames run eagerly-through-the-fixed-buffers before capture

    public GraphStream(IBackend backend, int hiddenSize, int rows = 1)
    {
        _backend = backend;
        InEmbed = new Tensor(new TensorShape(1, rows, hiddenSize), DType.F32);
        OutHidden = new Tensor(new TensorShape(1, rows, hiddenSize), DType.F32);
        DevicePos = backend.AllocDevicePos();
    }

    public void Dispose()
    {
        // Cleanup is best-effort: these run inside finally/using during exception unwind, and a CUDA
        // failure mid-generation leaves the context poisoned so the frees themselves throw
        // CUDA_ERROR_INVALID_VALUE — which then REPLACES the original exception (this masked the real
        // 12 GB-card failure in the 2026-07-24 HeartMuLa investigation). Log and continue instead.
        try { if (Graph is not null) { _backend.DisposeGraph(Graph); Graph = null; } }
        catch (Exception ex) { Logs.Warning($"[GraphStream] Graph dispose failed during cleanup (continuing): {ex.Message}"); }
        try { if (DevicePos != 0) _backend.FreeDevicePos(DevicePos); }
        catch (Exception ex) { Logs.Warning($"[GraphStream] DevicePos free failed during cleanup (continuing): {ex.Message}"); }
        try { InEmbed.Dispose(); }
        catch (Exception ex) { Logs.Warning($"[GraphStream] InEmbed dispose failed during cleanup (continuing): {ex.Message}"); }
        try { OutHidden.Dispose(); }
        catch (Exception ex) { Logs.Warning($"[GraphStream] OutHidden dispose failed during cleanup (continuing): {ex.Message}"); }
    }
}
