using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Layout;
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
/// buffer capacity, the rotary tables of every slot, and the sliding window the attention kernel applies.</summary>
internal sealed unsafe class VoiceIsolatorStreamAttention
{
    private const int FirstSlots = 64;

    private readonly float[] _cos;
    private readonly float[] _sin;
    private readonly float[] _negSin;

    public VoiceIsolatorStreamAttention(int cacheFrames, int capacity, int window, int headDim)
    {
        CacheFrames = cacheFrames;
        Capacity = capacity;
        // Each query sees at most CacheFrames keys, its own included, and the checkpoint's window when that is shorter.
        SlidingWindow = window > 0 ? Math.Min(window, cacheFrames) : cacheFrames;
        (_cos, _sin) = VoiceIsolatorOps.RopeTables(capacity, headDim);
        _negSin = Array.ConvertAll(_sin, s => -s);
    }

    public int CacheFrames { get; }

    public int Capacity { get; }

    /// <summary>Keys a query sees, itself included: the window the attention kernel applies.</summary>
    public int SlidingWindow { get; }

    public float[] Cos => _cos;

    public float[] Sin => _sin;

    /// <summary>Attention of the new frames <paramref name="q"/>, <paramref name="k"/>, <paramref name="v"/>
    /// (<c>[bands, heads, count, hd]</c>, unrotated) over the lane's cached past; writes their keys and values into the lane and
    /// returns the head-major output.</summary>
    public Tensor Attend(IBackend backend, VoiceIsolatorAttentionLane lane, Tensor q, Tensor k, Tensor v, int bands, int heads, int count, int hd)
    {
        if (lane.End + count > Capacity)
        {
            Compact(backend, lane, bands, heads, hd);
        }
        Reserve(backend, lane, lane.End + count, bands, heads, hd);
        int first = lane.End;
        VoiceIsolatorOps.Rope(backend, q, count, hd, first, _cos, _sin);
        VoiceIsolatorOps.Rope(backend, k, count, hd, first, _cos, _sin);
        // The new keys and values land in their slots on the backend: the cache stays where the attention reads it.
        Place(backend, lane.Keys!, k, first);
        Place(backend, lane.Values!, v, first);
        lane.End = first + count;
        // The cached buffers are attended in place: rows [0, first + count) are valid, query row i sits at first + i, and the
        // sliding window keeps each query to the last SlidingWindow keys. No past is copied.
        Tensor attended = new(new TensorShape(bands, heads, count, hd), DType.F32);
        AttentionCall call = AttentionCall.FromFlashAttention(attended, q, lane.Keys!, lane.Values!, first + count, causal: true,
            qOffset: first, scale: 1f / MathF.Sqrt(hd), softcap: 0f, sink: null, slidingWindow: SlidingWindow, alibiSlopes: null);
        backend.Attention(in call);
        return attended;
    }

    private void Reserve(IBackend backend, VoiceIsolatorAttentionLane lane, int slots, int bands, int heads, int hd)
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
            Move(backend, keys, lane.Keys!, lane.End);
            Move(backend, values, lane.Values!, lane.End);
        }
        lane.Dispose();
        lane.Keys = keys;
        lane.Values = values;
    }

    /// <summary>Moves the last <see cref="CacheFrames"/> keys and values to the front, the keys rotated back by the distance they moved.</summary>
    private void Compact(IBackend backend, VoiceIsolatorAttentionLane lane, int bands, int heads, int hd)
    {
        int keep = Math.Min(lane.End, CacheFrames);
        int shift = lane.End - keep;
        if (shift == 0)
        {
            return;
        }
        // The kept slots are gathered into fresh buffers, their keys rotate back by the distance they moved (every row by the same
        // angle, so the negated sine table at that position), and they are written back to the front of the cache.
        Tensor keys = new(new TensorShape(bands, heads, keep, hd), DType.F32);
        Tensor values = new(new TensorShape(bands, heads, keep, hd), DType.F32);
        StridedView kept = StridedView.Of(lane.Keys!.Shape).Slice(2, shift, keep);
        StridedView keptValues = StridedView.Of(lane.Values!.Shape).Slice(2, shift, keep);
        backend.CopyStrided(keys, lane.Keys, new StridedCopySpec(StridedView.Of(keys.Shape), kept));
        backend.CopyStrided(values, lane.Values, new StridedCopySpec(StridedView.Of(values.Shape), keptValues));
        VoiceIsolatorOps.RopeFixed(backend, keys, keep, hd, shift, _cos, _negSin);
        Place(backend, lane.Keys, keys, 0);
        Place(backend, lane.Values, values, 0);
        keys.Dispose();
        values.Dispose();
        lane.End = keep;
    }

    /// <summary>Writes <paramref name="src"/> <c>[bands, heads, count, hd]</c> into slots <c>[first, first + count)</c> of a cache buffer.</summary>
    private static void Place(IBackend backend, Tensor buffer, Tensor src, int first)
    {
        StridedView slots = StridedView.Of(buffer.Shape).Slice(2, first, src.Shape[2]);
        backend.CopyStrided(buffer, src, new StridedCopySpec(slots, StridedView.Of(src.Shape)));
    }

    /// <summary>Copies the first <paramref name="used"/> slots of <paramref name="from"/> into the same slots of <paramref name="to"/>.</summary>
    private static void Move(IBackend backend, Tensor to, Tensor from, int used)
    {
        backend.CopyStrided(to, from, new StridedCopySpec(StridedView.Of(to.Shape).Slice(2, 0, used), StridedView.Of(from.Shape).Slice(2, 0, used)));
    }
}
