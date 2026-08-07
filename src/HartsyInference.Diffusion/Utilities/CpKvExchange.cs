using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Per-block self-attention K/V rendezvous for 2 context-parallel ranks. Each exchange is a two-phase
/// <see cref="Barrier"/> protocol: (1) each rank host-materializes its post-RoPE local K/V <c>[Sr, dim]</c> on its
/// OWN thread (the D2H sync must run with that thread's ambient backend binding) and publishes the raw host
/// pointers; barrier; (2) each rank assembles kFull/vFull <c>[S, dim]</c> in global token order by host memcpy from
/// both slots; barrier — the second barrier guarantees neither rank disposes its locals while the peer still reads
/// them. Host-staged by design: the target box has no P2P, and publishing pointers keeps every device/stream touch
/// on the tensor's owning thread.
///
/// <para><b>Replicated prefix (joint-attention models).</b> Qwen-Image-style MMDiT ranks carry the txt stream
/// REPLICATED (every rank computes it deterministically) followed by their local img rows, so their local K/V is
/// <c>[prefix + Sr, dim]</c>. Passing <c>replicatedPrefixRows &gt; 0</c> assembles <c>[prefix + S, dim]</c> with the
/// prefix rows taken from the rank's OWN slot and only the suffix rows exchanged. Both ranks must pass the same
/// prefix (validated at the rendezvous). Wan's separate-cross-attention shape is the <c>prefix = 0</c> case.</para>
///
/// <para><b>Failure invariant.</b> A rank that throws ANYWHERE while its peer may still reach a barrier (inside an
/// exchange or anywhere else in its forward) MUST call <see cref="Abort"/> before unwinding — the waiting peer's
/// poll loop then throws instead of deadlocking. The pipeline's rank thunks wrap the whole forward in a catch that
/// aborts; a failed instance stays poisoned (one instance per generation).</para></summary>
public sealed unsafe class CpKvExchange : IDisposable
{
    private readonly Barrier _barrier = new(2);
    private readonly CpSequencePlan _plan;
    private readonly (nint K, nint V, int Prefix)[] _slots = new (nint, nint, int)[2];
    private volatile bool _failed;

    public CpKvExchange(CpSequencePlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        if (plan.Ranks.Count != 2)
        {
            throw new ArgumentException("CpKvExchange v1 supports exactly 2 ranks.", nameof(plan));
        }
        _plan = plan;
    }

    /// <summary>Marks the exchange failed so the peer's next (or current) barrier wait throws instead of hanging.</summary>
    public void Abort() => _failed = true;

    /// <summary>Trades this rank's local K/V <c>[(prefix +) Sr, dim]</c> for the assembled full-sequence pair
    /// <c>[(prefix +) S, dim]</c> (a leading batch-1 dim on the locals is preserved on the result). Consumes
    /// (disposes) the locals after the peer has finished reading them.</summary>
    /// <param name="replicatedPrefixRows">Leading rows every rank computed identically (a joint-attention txt
    /// stream); taken from this rank's own slot, never exchanged. Must match the peer's value.</param>
    public (Tensor KFull, Tensor VFull) Exchange(int rank, Tensor kLocal, Tensor vLocal, int replicatedPrefixRows = 0)
    {
        int dim = (int)kLocal.Shape[kLocal.Shape.Rank - 1];
        long expectedRows = replicatedPrefixRows + _plan.Ranks[rank].TokenCount;
        try
        {
            if (kLocal.DType != DType.F32 || vLocal.DType != DType.F32)
                throw new ArgumentException("CpKvExchange assembles F32 rows only.");
            if (kLocal.ElementCount != expectedRows * dim || vLocal.ElementCount != expectedRows * dim)
                throw new ArgumentException(
                    $"local K/V must be [{replicatedPrefixRows} prefix + {_plan.Ranks[rank].TokenCount} rank rows, {dim}]; " +
                    $"got {kLocal.Shape} / {vLocal.Shape}.");
            // Owning-thread D2H: after this the rows are plain host memory the peer can memcpy without touching
            // this rank's backend or stream.
            _slots[rank] = ((nint)kLocal.DataPointer, (nint)vLocal.DataPointer, replicatedPrefixRows);
        }
        catch
        {
            _failed = true;   // unblocks the peer's poll loop; no signal needed
            throw;
        }
        Rendezvous();   // both slots published
        Tensor? kFull = null, vFull = null;
        try
        {
            if (_slots[1 - rank].Prefix != replicatedPrefixRows)
                throw new InvalidOperationException(
                    $"replicatedPrefixRows mismatch across ranks ({replicatedPrefixRows} vs {_slots[1 - rank].Prefix}).");
            bool leadingBatch = kLocal.Shape.Rank == 3 && kLocal.Shape[0] == 1;
            kFull = Assemble(dim, replicatedPrefixRows, rank, leadingBatch, takeV: false);
            vFull = Assemble(dim, replicatedPrefixRows, rank, leadingBatch, takeV: true);
        }
        catch
        {
            _failed = true;
            kFull?.Dispose();
            vFull?.Dispose();
            throw;
        }
        Rendezvous();   // peer done reading our host rows — locals safe to dispose
        kLocal.Dispose();
        vLocal.Dispose();
        return (kFull!, vFull!);
    }

    private Tensor Assemble(int dim, int prefixRows, int ownRank, bool leadingBatch, bool takeV)
    {
        long rows = prefixRows + _plan.TotalTokens;
        TensorShape shape = leadingBatch ? new TensorShape(1, rows, dim) : new TensorShape(rows, dim);
        Tensor full = new Tensor(shape, DType.F32);
        float* dst = (float*)full.DataPointer;
        if (prefixRows > 0)
        {
            float* own = (float*)(takeV ? _slots[ownRank].V : _slots[ownRank].K);
            long prefixBytes = (long)prefixRows * dim * 4;
            Buffer.MemoryCopy(own, dst, prefixBytes, prefixBytes);
        }
        for (int r = 0; r < 2; r++)
        {
            CpRankRange range = _plan.Ranks[r];
            float* src = (float*)(takeV ? _slots[r].V : _slots[r].K) + (long)prefixRows * dim;
            long bytes = (long)range.TokenCount * dim * 4;
            Buffer.MemoryCopy(src, dst + ((long)prefixRows + range.TokenStart) * dim, bytes, bytes);
        }
        return full;
    }

    private void Rendezvous()
    {
        // Timed wait so a peer that aborted WITHOUT signaling (it threw outside any barrier) still unblocks us;
        // Barrier rescinds this participant's signal on each timeout, so re-signaling next iteration is safe.
        while (!_barrier.SignalAndWait(100))
        {
            ThrowIfFailed();
        }
        ThrowIfFailed();
    }

    private void ThrowIfFailed()
    {
        if (_failed)
        {
            throw new InvalidOperationException("Context-parallel peer rank failed — aborting this rank's forward.");
        }
    }

    public void Dispose() => _barrier.Dispose();
}
