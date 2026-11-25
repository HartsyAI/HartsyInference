using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VoiceIsolator;

/// <summary>One lane's time-attention past for one attention block: keys (rotated once, at their slot) and values in buffers
/// <c>[bands, heads, capacity, headDim]</c>. A slot's index is its rotary position; <see cref="End"/> counts the slots in use.</summary>
internal sealed class VoiceIsolatorAttentionLane : IDisposable
{
    public Tensor? Keys { get; set; }

    public Tensor? Values { get; set; }

    public int End { get; set; }

    public void Dispose()
    {
        Keys?.Dispose();
        Values?.Dispose();
        Keys = null;
        Values = null;
    }
}

/// <summary>The settings every attention block of a stream shares and the attention step itself: the past each query may see, the
/// buffer capacity, the rotary tables of every slot, and the checkpoint's own window bound.</summary>
internal sealed unsafe class VoiceIsolatorStreamAttention
{
    private const int FirstSlots = 64;
    private const float MaskedScore = -1e9f;

    private readonly float[] _cos;
    private readonly float[] _sin;

    public VoiceIsolatorStreamAttention(int cacheFrames, int capacity, int window, int headDim)
    {
        CacheFrames = cacheFrames;
        Capacity = capacity;
        Window = window;
        (_cos, _sin) = VoiceIsolatorOps.RopeTables(capacity, headDim);
    }

    public int CacheFrames { get; }

    public int Capacity { get; }

    public int Window { get; }

    public float[] Cos => _cos;

    public float[] Sin => _sin;

    /// <summary>Attention of the new frames <paramref name="q"/>, <paramref name="k"/>, <paramref name="v"/>
    /// (<c>[bands, heads, count, hd]</c>, unrotated) over the lane's cached past; writes their keys and values into the lane and
    /// returns the head-major output.</summary>
    public Tensor Attend(IBackend backend, VoiceIsolatorAttentionLane lane, Tensor q, Tensor k, Tensor v, int bands, int heads, int count, int hd)
    {
        if (lane.End + count > Capacity)
        {
            Compact(lane, bands, heads, hd);
        }
        Reserve(lane, lane.End + count, bands, heads, hd);
        int first = lane.End;
        VoiceIsolatorOps.Rope(backend, q, count, hd, first, _cos, _sin);
        VoiceIsolatorOps.Rope(backend, k, count, hd, first, _cos, _sin);
        int held = (int)lane.Keys!.Shape[2];
        float* kc = (float*)lane.Keys.DataPointer;
        float* vc = (float*)lane.Values!.DataPointer;
        float* ks = (float*)k.DataPointer;
        float* vs = (float*)v.DataPointer;
        for (int bh = 0; bh < bands * heads; bh++)
        {
            Buffer.MemoryCopy(ks + (long)bh * count * hd, kc + ((long)bh * held + first) * hd, (long)count * hd * 4, (long)count * hd * 4);
            Buffer.MemoryCopy(vs + (long)bh * count * hd, vc + ((long)bh * held + first) * hd, (long)count * hd * 4, (long)count * hd * 4);
        }
        lane.End = first + count;
        int past = Math.Min(first, CacheFrames);
        int total = past + count;
        Tensor keys = new(new TensorShape(bands, heads, total, hd), DType.F32);
        Tensor values = new(new TensorShape(bands, heads, total, hd), DType.F32);
        float* kd = (float*)keys.DataPointer;
        float* vd = (float*)values.DataPointer;
        for (int bh = 0; bh < bands * heads; bh++)
        {
            Buffer.MemoryCopy(kc + ((long)bh * held + first - past) * hd, kd + (long)bh * total * hd, (long)total * hd * 4, (long)total * hd * 4);
            Buffer.MemoryCopy(vc + ((long)bh * held + first - past) * hd, vd + (long)bh * total * hd, (long)total * hd * 4, (long)total * hd * 4);
        }
        Tensor mask = new(new TensorShape(1, 1, count, total), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int i = 0; i < count; i++)
        {
            int query = past + i;
            for (int j = 0; j < total; j++)
            {
                bool visible = j <= query && (Window == 0 || j > query - Window);
                mp[(long)i * total + j] = visible ? 0f : MaskedScore;
            }
        }
        Tensor attended = new(new TensorShape(bands, heads, count, hd), DType.F32);
        backend.ScaledDotProductAttention(attended, q, keys, values, mask, 1f / MathF.Sqrt(hd));
        keys.Dispose();
        values.Dispose();
        mask.Dispose();
        return attended;
    }

    private void Reserve(VoiceIsolatorAttentionLane lane, int slots, int bands, int heads, int hd)
    {
        int held = lane.Keys is null ? 0 : (int)lane.Keys.Shape[2];
        if (slots <= held)
        {
            return;
        }
        int size = Math.Min(Capacity, Math.Max(slots, Math.Max(2 * held, FirstSlots)));
        Tensor keys = new(new TensorShape(bands, heads, size, hd), DType.F32);
        Tensor values = new(new TensorShape(bands, heads, size, hd), DType.F32);
        if (lane.End > 0)
        {
            for (int bh = 0; bh < bands * heads; bh++)
            {
                long used = (long)lane.End * hd * 4;
                Buffer.MemoryCopy((float*)lane.Keys!.DataPointer + (long)bh * held * hd, (float*)keys.DataPointer + (long)bh * size * hd, used, used);
                Buffer.MemoryCopy((float*)lane.Values!.DataPointer + (long)bh * held * hd, (float*)values.DataPointer + (long)bh * size * hd, used, used);
            }
        }
        lane.Dispose();
        lane.Keys = keys;
        lane.Values = values;
    }

    /// <summary>Moves the last <see cref="CacheFrames"/> keys and values to the front, the keys rotated back by the distance they moved.</summary>
    private void Compact(VoiceIsolatorAttentionLane lane, int bands, int heads, int hd)
    {
        int keep = Math.Min(lane.End, CacheFrames);
        int shift = lane.End - keep;
        if (shift == 0)
        {
            return;
        }
        int half = hd / 2;
        int held = (int)lane.Keys!.Shape[2];
        float* kc = (float*)lane.Keys.DataPointer;
        float* vc = (float*)lane.Values!.DataPointer;
        for (int bh = 0; bh < bands * heads; bh++)
        {
            float* keys = kc + (long)bh * held * hd;
            float* values = vc + (long)bh * held * hd;
            for (int s = 0; s < keep; s++)
            {
                float* src = keys + (long)(shift + s) * hd;
                float* dst = keys + (long)s * hd;
                for (int i = 0; i < half; i++)
                {
                    float c = _cos[(long)shift * half + i];
                    float sn = -_sin[(long)shift * half + i];
                    float x1 = src[i];
                    float x2 = src[half + i];
                    dst[i] = x1 * c - x2 * sn;
                    dst[half + i] = x1 * sn + x2 * c;
                }
                Buffer.MemoryCopy(values + (long)(shift + s) * hd, values + (long)s * hd, hd * 4L, hd * 4L);
            }
        }
        lane.End = keep;
    }
}
