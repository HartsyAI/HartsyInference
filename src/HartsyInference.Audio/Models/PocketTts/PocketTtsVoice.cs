using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.PocketTts;

/// <summary>A Pocket-TTS predefined voice: the FlowLM transformer's per-layer self-attention <b>KV cache</b>
/// after the reference clip was primed through it (moshi streaming state). Each layer ships as
/// <c>transformer.layers.{i}.self_attn/cache</c> <c>[2, 1, P, heads, headDim]</c> (0=K, 1=V, already RoPE'd at
/// positions 0..P-1) plus an <c>.../offset</c> = P. Reshaped here to the backbone's attention layout
/// <c>[1, heads, P, headDim]</c> per layer so generation can attend over <c>[voice ++ text ++ latents]</c>.
///
/// <para>The voice is REQUIRED — with no voice the model end-of-sequences immediately (near-silence). Predefined
/// voices live in the ungated repo under <c>languages/&lt;lang&gt;/embeddings/&lt;name&gt;.safetensors</c>.</para></summary>
public sealed unsafe class PocketTtsVoice : IDisposable
{
    public Tensor[] PrefixK { get; }
    public Tensor[] PrefixV { get; }
    public int PrefixLen { get; }
    private int _disposed;

    private PocketTtsVoice(Tensor[] k, Tensor[] v, int prefixLen)
    {
        PrefixK = k;
        PrefixV = v;
        PrefixLen = prefixLen;
    }

    /// <summary>Builds a voice from a loaded voice-state safetensors dict (keys
    /// <c>transformer.layers.{i}.self_attn/cache</c>).</summary>
    public static PocketTtsVoice Load(IReadOnlyDictionary<string, Tensor> w, int layers = 6, int heads = 16, int headDim = 64)
    {
        Tensor[] k = new Tensor[layers];
        Tensor[] v = new Tensor[layers];
        int prefixLen = 0;
        for (int i = 0; i < layers; i++)
        {
            string key = $"transformer.layers.{i}.self_attn/cache";
            if (!w.TryGetValue(key, out Tensor? cacheRef))
                throw new ArgumentException($"voice state missing '{key}'.");
            Tensor cache = WhisperOps.EnsureF32(cacheRef);   // [2,1,P,heads,headDim]
            int p = (int)cache.Shape[2];
            prefixLen = p;
            k[i] = SliceTranspose(cache, 0, p, heads, headDim);
            v[i] = SliceTranspose(cache, 1, p, heads, headDim);
        }
        return new PocketTtsVoice(k, v, prefixLen);
    }

    // cache[kv, 0, s, h, d] -> out[0, h, s, d]  ([1, heads, P, headDim])
    private static Tensor SliceTranspose(Tensor cache, int kv, int p, int heads, int headDim)
    {
        Tensor outT = new(new TensorShape(1, heads, p, headDim), DType.F32);
        float* src = (float*)cache.DataPointer;
        float* dst = (float*)outT.DataPointer;
        long kvStride = (long)p * heads * headDim;   // stride of dim 0 (batch=1)
        for (int s = 0; s < p; s++)
            for (int h = 0; h < heads; h++)
            {
                long sBase = kv * kvStride + ((long)s * heads + h) * headDim;
                long dBase = ((long)h * p + s) * headDim;
                for (int d = 0; d < headDim; d++) dst[dBase + d] = src[sBase + d];
            }
        return outT;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in PrefixK) t.Dispose();
        foreach (Tensor t in PrefixV) t.Dispose();
    }
}
