namespace HartsyInference.Audio.Models.LanguageModels.Gpt;

/// <summary>Per-layer key/value cache for incremental <see cref="GptBackbone"/> decoding. Keys/values are kept
/// on the host in token-major <c>[t, hidden]</c> layout (the pre-multi-head shape the blocks produce), so the
/// single-token attention runs on CPU over the cached prefix while the projections stay on the backend — the
/// growing K/V never crosses PCIe. One <see cref="GptKvLayer"/> per block; <see cref="Length"/> is the shared
/// number of cached positions (kept in lock-step across layers by the backbone).</summary>
public sealed class GptKvCache
{
    /// <summary>K/V rows for one block. <c>K</c>/<c>V</c> are <c>[capacity * hidden]</c> flat, row <c>t</c> at
    /// offset <c>t * hidden</c>; head <c>i</c> of a row is the contiguous slice <c>[i*headDim, (i+1)*headDim)</c>.</summary>
    public sealed class GptKvLayer(int capacity, int hidden)
    {
        public readonly float[] K = new float[checked(capacity * hidden)];
        public readonly float[] V = new float[checked(capacity * hidden)];
    }

    public GptKvCache(int numLayers, int capacity, int hidden)
    {
        Capacity = capacity;
        Hidden = hidden;
        Layers = new GptKvLayer[numLayers];
        for (int i = 0; i < numLayers; i++) Layers[i] = new GptKvLayer(capacity, hidden);
    }

    public GptKvLayer[] Layers { get; }

    /// <summary>Number of positions currently cached (identical across layers).</summary>
    public int Length { get; internal set; }

    /// <summary>Maximum positions the cache can hold (the model's block size).</summary>
    public int Capacity { get; }

    public int Hidden { get; }

    /// <summary>Empties the cache for a fresh sequence without reallocating.</summary>
    public void Reset() => Length = 0;
}
