using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Mimi;

/// <summary>Per-utterance carried state for <see cref="MimiSeanetDecoder"/>'s streaming decode: the last <c>kernel-1</c> input samples of every causal conv, plus the un-emitted overlap-add tail of every upsample <c>ConvTranspose1d</c> stage.</summary>
/// <remarks>Slots are keyed positionally in call order (in-conv, per-stage r1, out-conv for
/// conv history; per-stage upsample for transpose history) — <see cref="MimiSeanetDecoder"/> always touches them
/// in the same fixed order, so a simple index counter is enough; no dictionary needed.
///
/// <para>All slots start null, which reads as "no history yet" — the same as the non-streaming path's implicit
/// zero-padding — so the very first chunk of an utterance produces byte-identical output to today's
/// <see cref="MimiSeanetDecoder.Forward"/> without any special-casing.</para></remarks>
internal sealed unsafe class MimiSeanetDecoderStreamState : IDisposable
{
    private readonly Tensor?[] _convTail = new Tensor?[6];      // in-conv, r1 x4, out-conv (r2 is kernel=1, needs none)
    // 5 slots, not 4: slot 4 is Mimi.DecodeStreaming's own pre-transformer depthwise upsample conv (12.5->25 Hz),
    // which has the exact same crop-and-discard tail problem as the 4 SEANet decoder upsample stages below it —
    // reused here rather than duplicating the overlap-add logic for a 5th call site.
    private readonly Tensor?[] _transposeTail = new Tensor?[5];
    private int _disposed;

    /// <summary>Prepends <paramref name="kernel"/><c>-1</c> samples of left context to <paramref name="input"/> along the time axis — the carried tail from the previous chunk, or zeros on an utterance's first chunk.</summary>
    /// <remarks>Returns a tensor the caller owns and must dispose; also updates the stored tail to the
    /// last <c>kernel-1</c> samples of the AUGMENTED (context+input) sequence — not of <paramref name="input"/>
    /// alone, which would be wrong when <paramref name="t"/> is smaller than the history length, since then part
    /// of the next chunk's context must still come from what was carried into THIS call.</remarks>
    public Tensor Augment(int slot, Tensor input, int inDim, int t, int kernel, int batch)
    {
        ThrowIfDisposed();
        int histLen = kernel - 1;
        if (histLen == 0)
        {
            return input; // kernel=1: nothing to carry, caller must not dispose (same object returned)
        }
        Tensor? tail = _convTail[slot];
        int totalLen = histLen + t;
        Tensor augmented = new(new TensorShape(batch, inDim, totalLen), DType.F32);
        float* dst = (float*)augmented.DataPointer;
        float* src = (float*)input.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long dstBase = (long)b * inDim * totalLen;
            if (tail is not null)
            {
                float* tp = (float*)tail.DataPointer;
                long tailBase = (long)b * inDim * histLen;
                for (int c = 0; c < inDim; c++)
                {
                    Buffer.MemoryCopy(tp + tailBase + (long)c * histLen, dst + dstBase + (long)c * totalLen, (long)histLen * 4, (long)histLen * 4);
                }
            }
            else
            {
                new Span<float>(dst + dstBase, inDim * histLen).Clear();
            }
            long srcBase = (long)b * inDim * t;
            for (int c = 0; c < inDim; c++)
            {
                Buffer.MemoryCopy(src + srcBase + (long)c * t, dst + dstBase + (long)c * totalLen + histLen, (long)t * 4, (long)t * 4);
            }
        }

        // New tail = last histLen samples of the AUGMENTED sequence (context+input), so a chunk smaller than
        // histLen still carries forward whatever earlier context it didn't fully displace.
        Tensor newTail = new(new TensorShape(batch, inDim, histLen), DType.F32);
        float* ntp = (float*)newTail.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long ntBase = (long)b * inDim * histLen;
            long dstBase = (long)b * inDim * totalLen;
            for (int c = 0; c < inDim; c++)
            {
                Buffer.MemoryCopy(dst + dstBase + (long)c * totalLen + (totalLen - histLen), ntp + ntBase + (long)c * histLen, (long)histLen * 4, (long)histLen * 4);
            }
        }
        tail?.Dispose();
        _convTail[slot] = newTail;
        return augmented;
    }

    /// <summary>Adds this stage's carried overlap tail into the first <c>overlap</c> samples of <paramref name="rawFull"/> (the UNCROPPED transpose-conv output, length <c>tUp+overlap</c>) in place, then splits it into the emitted chunk plus the new carried tail; disposes <paramref name="rawFull"/>.</summary>
    /// <remarks>This is the overlap-add reconstruction of what a single monolithic
    /// <c>ConvTranspose1d</c> call over the whole utterance would have produced, split across chunk boundaries.</remarks>
    public Tensor SplitTransposeOutput(int slot, Tensor rawFull, int outCh, int tUp, int overlap, int batch)
    {
        ThrowIfDisposed();
        float* raw = (float*)rawFull.DataPointer;
        Tensor? tail = _transposeTail[slot];
        if (tail is not null)
        {
            float* tp = (float*)tail.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                long rawBase = (long)b * outCh * (tUp + overlap);
                long tailBase = (long)b * outCh * overlap;
                for (int c = 0; c < outCh; c++)
                {
                    long rowRaw = rawBase + (long)c * (tUp + overlap);
                    long rowTail = tailBase + (long)c * overlap;
                    for (int i = 0; i < overlap; i++)
                    {
                        raw[rowRaw + i] += tp[rowTail + i];
                    }
                }
            }
            tail.Dispose();
        }

        Tensor emitted = new(new TensorShape(batch, outCh, tUp), DType.F32);
        Tensor newTail = new(new TensorShape(batch, outCh, overlap), DType.F32);
        float* ep = (float*)emitted.DataPointer;
        float* ntp = (float*)newTail.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long rawBase = (long)b * outCh * (tUp + overlap);
            long emitBase = (long)b * outCh * tUp;
            long tailBase = (long)b * outCh * overlap;
            for (int c = 0; c < outCh; c++)
            {
                Buffer.MemoryCopy(raw + rawBase + (long)c * (tUp + overlap), ep + emitBase + (long)c * tUp, (long)tUp * 4, (long)tUp * 4);
                Buffer.MemoryCopy(raw + rawBase + (long)c * (tUp + overlap) + tUp, ntp + tailBase + (long)c * overlap, (long)overlap * 4, (long)overlap * 4);
            }
        }
        rawFull.Dispose();
        _transposeTail[slot] = newTail;
        return emitted;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(MimiSeanetDecoderStreamState));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor? t in _convTail) t?.Dispose();
        foreach (Tensor? t in _transposeTail) t?.Dispose();
    }
}
