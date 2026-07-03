namespace HartsyInference.Audio.Models.Music;

/// <summary>Per-layer key/value cache for incremental <see cref="MusicGenDecoder"/> decoding. Self-attn
/// keys/values are kept on the host in token-major <c>[t, hidden]</c> layout (the pre-multi-head shape the
/// blocks produce), so the single-token attention runs on CPU over the cached prefix while the projections
/// stay on the backend — the growing K/V never crosses PCIe. Cross-attn K/V depend only on the text states,
/// so they are projected once at cache creation (<see cref="MusicGenDecoder.CreateCache"/>) and reused every
/// step. One <see cref="MusicGenKvLayer"/> per block; <see cref="Length"/> is the shared number of cached
/// positions (kept in lock-step across layers by the decoder).</summary>
public sealed class MusicGenKvCache
{
    /// <summary>K/V rows for one block. <c>K</c>/<c>V</c> are <c>[capacity * hidden]</c> flat, row <c>t</c> at
    /// offset <c>t * hidden</c>; head <c>i</c> of a row is the contiguous slice <c>[i*headDim, (i+1)*headDim)</c>.
    /// <c>CrossK</c>/<c>CrossV</c> are the once-projected text K/V, <c>[tText * hidden]</c> in the same layout.</summary>
    public sealed class MusicGenKvLayer(int capacity, int hidden)
    {
        public readonly float[] K = new float[checked(capacity * hidden)];
        public readonly float[] V = new float[checked(capacity * hidden)];
        public float[] CrossK = [];
        public float[] CrossV = [];
    }

    public MusicGenKvCache(int numLayers, int capacity, int hidden)
    {
        Capacity = capacity;
        Hidden = hidden;
        Layers = new MusicGenKvLayer[numLayers];
        for (int i = 0; i < numLayers; i++) Layers[i] = new MusicGenKvLayer(capacity, hidden);
    }

    public MusicGenKvLayer[] Layers { get; }

    /// <summary>Number of positions currently cached (identical across layers).</summary>
    public int Length { get; internal set; }

    /// <summary>Maximum positions the cache can hold (the caller sizes it to the generation length).</summary>
    public int Capacity { get; }

    public int Hidden { get; }

    /// <summary>Text-state length (rows in <c>CrossK</c>/<c>CrossV</c>), set at cache creation.</summary>
    public int CrossLength { get; internal set; }

    /// <summary>Empties the self-attn cache for a fresh sequence without reallocating (cross K/V stay).</summary>
    public void Reset() => Length = 0;
}
