using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Music;

/// <summary>Per-layer key/value cache for incremental <see cref="MusicGenDecoder"/> decoding, fully device-resident so the single-token step never crosses PCIe.</summary>
/// <remarks>Self-attn K/V are appended in place on the
/// backend (<see cref="HartsyInference.Core.Backends.IBackend.KvCacheAppend"/>) into a fixed head-major
/// <c>[1, heads, capacity, headDim]</c> buffer, then attended with <see cref="HartsyInference.Core.Backends.IBackend.FlashAttention"/>
/// over the valid prefix (kvLen told separately) — no host round-trip, no O(n²) concat. Cross-attn K/V depend
/// only on the text states, so they are projected once at cache creation (<see cref="MusicGenDecoder.CreateCache"/>)
/// into device tensors and reused every step. One <see cref="MusicGenKvLayer"/> per block; <see cref="Length"/>
/// is the shared number of cached positions (kept in lock-step across layers by the decoder).</remarks>
public sealed class MusicGenKvCache : IDisposable
{
    /// <summary>Device K/V buffers for one block, head-major <c>[1, heads, capacity, headDim]</c>; valid keys are the first <see cref="Length"/> positions (FlashAttention is told the count). <c>CrossK</c>/<c>CrossV</c> are the once-projected text K/V, head-major <c>[1, heads, crossLen, headDim]</c>.</summary>
    public sealed class MusicGenKvLayer : IDisposable
    {
        public readonly Tensor K;
        public readonly Tensor V;
        public Tensor? CrossK;
        public Tensor? CrossV;

        public MusicGenKvLayer(int batch, int numHeads, int capacity, int headDim)
        {
            K = new Tensor(new TensorShape(batch, numHeads, capacity, headDim), DType.F32);
            V = new Tensor(new TensorShape(batch, numHeads, capacity, headDim), DType.F32);
        }

        public void Dispose()
        {
            K.Dispose();
            V.Dispose();
            CrossK?.Dispose();
            CrossV?.Dispose();
        }
    }

    private int _disposed;

    /// <summary>Batch size: 1 (no CFG) or 2 (CFG cond+uncond decoded together). Element 0 is the conditional stream, element 1 the unconditional (null-cross) stream.</summary>
    public int Batch { get; }

    public MusicGenKvCache(int batch, int numLayers, int numHeads, int capacity, int headDim)
    {
        Batch = batch;
        Capacity = capacity;
        NumHeads = numHeads;
        HeadDim = headDim;
        Layers = new MusicGenKvLayer[numLayers];
        for (int i = 0; i < numLayers; i++) Layers[i] = new MusicGenKvLayer(batch, numHeads, capacity, headDim);
    }

    public MusicGenKvLayer[] Layers { get; }

    /// <summary>Number of positions currently cached (identical across layers).</summary>
    public int Length { get; internal set; }

    /// <summary>Maximum positions the cache can hold (the caller sizes it to the generation length).</summary>
    public int Capacity { get; }

    public int NumHeads { get; }

    public int HeadDim { get; }

    /// <summary>Text-state length (valid rows in <c>CrossK</c>/<c>CrossV</c>), set at cache creation.</summary>
    public int CrossLength { get; internal set; }

    /// <summary>Device buffer handle {kvLen, qOffset} for graph-replayable decode (0 = eager / host position). The self-attn KV-append + flash kernels read this instead of a host int so one captured graph replays each step.</summary>
    public ulong DevicePos { get; internal set; }

    /// <summary>Backend that owns <see cref="DevicePos"/> (used to free it on dispose).</summary>
    internal IBackend? PosBackend { get; set; }

    /// <summary>Empties the self-attn cache for a fresh sequence without reallocating (cross K/V stay).</summary>
    public void Reset() => Length = 0;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (MusicGenKvLayer l in Layers) l.Dispose();
        if (DevicePos != 0) { PosBackend?.FreeDevicePos(DevicePos); DevicePos = 0; }
    }
}
